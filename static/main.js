/**
 * Sign Language Recognition — Frontend Logic
 * ============================================
 * - MediaPipe Holistic: trích xuất keypoints từ webcam
 * - Gửi 30 frames × 1662 features về server /predict
 * - Hiển thị kết quả realtime + tích hợp Gemini AI sinh câu
 */

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS & CONFIG
// ═══════════════════════════════════════════════════════════════════════════════

const SEQ_LEN   = 30;       // Số frame cho 1 lần dự đoán
const THRESHOLD = 0.80;     // Ngưỡng tin cậy 80%

// ═══════════════════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════════════════

let sequence        = [];        // Buffer chứa frames
let detectedWords   = [];        // Danh sách từ đã nhận diện
let lastPrediction  = '';        // Từ dự đoán gần nhất (tránh trùng liên tục)
let predicting      = false;     // Đang gửi request predict
let isPaused        = false;     // Trạng thái tạm dừng nhận diện
let noHandFrames    = 0;         // Đếm số frame liên tiếp không có tay
let latestKeypoints = null;      // Lưu tọa độ mới nhất để đồng bộ FPS
let predictionsList = [];        // Lưu lịch sử các lần dự đoán gần nhất để lọc nhiễu
let lastGeneratedWords = [];     // Theo dõi từ đã sinh câu

// ═══════════════════════════════════════════════════════════════════════════════
// DOM ELEMENTS
// ═══════════════════════════════════════════════════════════════════════════════

const video      = document.getElementById('vid');
const canvas     = document.getElementById('cvs');
const ctx        = canvas.getContext('2d');
const detLabel   = document.getElementById('det-lbl');
const liveSent   = document.getElementById('live-sent');
const probList   = document.getElementById('prob-list');
const chipsEl    = document.getElementById('chips');
const aiBox      = document.getElementById('ai-box');
const dot        = document.getElementById('dot');
const statusText = document.getElementById('st');
const frameBar   = document.getElementById('fb');
const frameCount = document.getElementById('fc');
const pauseBtn   = document.getElementById('btn-pause');



// ═══════════════════════════════════════════════════════════════════════════════
// PAUSE / RESUME
// ═══════════════════════════════════════════════════════════════════════════════

if (pauseBtn) {
    pauseBtn.addEventListener('click', () => {
        isPaused = !isPaused;
        pauseBtn.textContent = isPaused ? '▶' : '⏸';
        pauseBtn.classList.toggle('active', isPaused);
        if (isPaused) {
            detLabel.textContent = '⏸️ Đã tạm dừng nhận diện';
        } else {
            detLabel.textContent = '— Chờ nhận diện —';
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// KEYPOINT EXTRACTION
// ═══════════════════════════════════════════════════════════════════════════════
// Trích xuất 1662 features GỐC (giống data thu thập trong notebook)
// để gửi lên server, server sẽ tự chọn lọc + chuẩn hóa.
// Format: pose(33×4=132) + face(468×3=1404) + lh(21×3=63) + rh(21×3=63) = 1662

function extractKeypoints(results) {
    // ── Pose: 33 landmarks × 4 values (x, y, z, visibility) = 132 ──
    const pose = new Array(132).fill(0);
    if (results.poseLandmarks) {
        for (let i = 0; i < results.poseLandmarks.length; i++) {
            const lm = results.poseLandmarks[i];
            pose[i * 4]     = lm.x;
            pose[i * 4 + 1] = lm.y;
            pose[i * 4 + 2] = lm.z;
            pose[i * 4 + 3] = lm.visibility || 0;
        }
    }

    // ── Face: 468 landmarks × 3 values (x, y, z) = 1404 ──
    const face = new Array(1404).fill(0);
    if (results.faceLandmarks) {
        for (let i = 0; i < results.faceLandmarks.length; i++) {
            const lm = results.faceLandmarks[i];
            face[i * 3]     = lm.x;
            face[i * 3 + 1] = lm.y;
            face[i * 3 + 2] = lm.z;
        }
    }

    // ── Left hand: 21 landmarks × 3 values = 63 ──
    const lh = new Array(63).fill(0);
    if (results.leftHandLandmarks) {
        for (let i = 0; i < results.leftHandLandmarks.length; i++) {
            const lm = results.leftHandLandmarks[i];
            lh[i * 3]     = lm.x;
            lh[i * 3 + 1] = lm.y;
            lh[i * 3 + 2] = lm.z;
        }
    }

    // ── Right hand: 21 landmarks × 3 values = 63 ──
    const rh = new Array(63).fill(0);
    if (results.rightHandLandmarks) {
        for (let i = 0; i < results.rightHandLandmarks.length; i++) {
            const lm = results.rightHandLandmarks[i];
            rh[i * 3]     = lm.x;
            rh[i * 3 + 1] = lm.y;
            rh[i * 3 + 2] = lm.z;
        }
    }

    // Ghép đúng thứ tự: pose | face | left_hand | right_hand = 1662
    return [...pose, ...face, ...lh, ...rh];
}

// ═══════════════════════════════════════════════════════════════════════════════
// DRAWING — Vẽ landmarks lên canvas
// ═══════════════════════════════════════════════════════════════════════════════

function drawResults(results) {
    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

    // Pose skeleton
    if (results.poseLandmarks) {
        drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS,
            { color: 'rgba(124, 58, 237, 0.6)', lineWidth: 2 });
        drawLandmarks(ctx, results.poseLandmarks,
            { color: 'rgba(6, 182, 212, 0.8)', lineWidth: 1, radius: 2 });
    }

    // Left hand
    if (results.leftHandLandmarks) {
        drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS,
            { color: 'rgba(239, 68, 68, 0.7)', lineWidth: 2 });
        drawLandmarks(ctx, results.leftHandLandmarks,
            { color: 'rgba(239, 68, 68, 0.9)', lineWidth: 1, radius: 3 });
    }

    // Right hand
    if (results.rightHandLandmarks) {
        drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS,
            { color: 'rgba(34, 197, 94, 0.7)', lineWidth: 2 });
        drawLandmarks(ctx, results.rightHandLandmarks,
            { color: 'rgba(34, 197, 94, 0.9)', lineWidth: 1, radius: 3 });
    }

    ctx.restore();
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROBABILITY DISPLAY — Hiển thị top 10 nhãn
// ═══════════════════════════════════════════════════════════════════════════════

function updateProbDisplay(top_k) {
    probList.innerHTML = '';
    top_k.forEach((item, i) => {
        const pct = (item.probability * 100).toFixed(1);
        const div = document.createElement('div');
        div.className = 'prob-item' + (item.probability > 0 && i === 0 ? ' active' : '');
        div.innerHTML = `
            <div class="prob-row">
                <span>${item.action}</span>
                <span class="pv">${pct}%</span>
            </div>
            <div class="track">
                <div class="fill" style="width:${item.probability * 100}%"></div>
            </div>
        `;
        probList.appendChild(div);
    });
}

function initProbDisplay() {
    // Lấy 10 nhãn đầu tiên làm mặc định ban đầu với xác suất 0%
    const initialActions = window.ACTIONS ? window.ACTIONS.slice(0, 10) : [];
    const initialTopK = initialActions.map(action => ({
        action: action,
        probability: 0
    }));
    updateProbDisplay(initialTopK);
}

// Khởi tạo hiển thị xác suất ngay lúc đầu
initProbDisplay();

// ═══════════════════════════════════════════════════════════════════════════════
// WORD CHIPS — Quản lý danh sách từ đã nhận diện
// ═══════════════════════════════════════════════════════════════════════════════

function addChip(word) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = word;
    chipsEl.appendChild(chip);

    // Cập nhật live sentence
    liveSent.textContent = detectedWords.join(' → ');
}


// ═══════════════════════════════════════════════════════════════════════════════
// GENERATE SENTENCE — Gửi từ cho Gemini AI
// ═══════════════════════════════════════════════════════════════════════════════

async function generateSentence() {
    if (detectedWords.length === 0) {
        aiBox.textContent = 'Chưa có từ nào để sinh câu!';
        return;
    }

    // Tránh gọi API liên tục nếu danh sách từ chưa có gì mới
    if (JSON.stringify(detectedWords) === JSON.stringify(lastGeneratedWords)) {
        return;
    }

    // Tạm dừng nhận dạng
    isPaused = true;
    if (pauseBtn) {
        pauseBtn.textContent = '▶';
        pauseBtn.classList.add('active');
    }
    detLabel.textContent = '⏸️ Đã tạm dừng nhận diện';

    const btn = document.getElementById('btn-gen');
    if (btn) btn.disabled = true;
    aiBox.classList.add('loading');
    aiBox.textContent = '✨ Đang sinh câu…';

    try {
        const res = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ words: detectedWords })
        });

        const data = await res.json();
        if (data.sentence && !data.sentence.includes("Lỗi") && !data.sentence.includes("API đang bị")) {
            // Chỉ cập nhật trạng thái nếu không phải thông báo lỗi từ server
            lastGeneratedWords = [...detectedWords];
        }
        aiBox.textContent = data.sentence || 'Không có kết quả.';
        aiBox.classList.remove('loading');
        liveSent.textContent = data.sentence || '…';
    } catch (err) {
        console.error('Generate error:', err);
        aiBox.textContent = '❌ Lỗi kết nối AI';
        aiBox.classList.remove('loading');
    }

    if (btn) btn.disabled = false;
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESET — Xóa toàn bộ
// ═══════════════════════════════════════════════════════════════════════════════

function resetAll() {
    sequence       = [];
    detectedWords  = [];
    lastPrediction = '';
    lastGeneratedWords = [];
    isPaused       = false;
    predictionsList = [];

    if (pauseBtn) {
        pauseBtn.textContent = '⏸';
        pauseBtn.classList.remove('active');
    }

    chipsEl.innerHTML  = '';
    probList.innerHTML = '';
    initProbDisplay(); // Khôi phục lại 10 nhãn 0%
    aiBox.textContent  = 'Nhấn "Sinh câu" sau khi có từ…';
    aiBox.classList.remove('loading');
    detLabel.textContent  = '— Chờ nhận diện —';
    liveSent.textContent  = '…';
    frameBar.style.width  = '0%';
    frameCount.textContent = `0 / ${SEQ_LEN} frames`;
}

// Gán vào window để HTML onclick có thể gọi
window.generateSentence = generateSentence;
window.resetAll          = resetAll;

// ═══════════════════════════════════════════════════════════════════════════════
// MEDIAPIPE HOLISTIC — Khởi tạo
// ═══════════════════════════════════════════════════════════════════════════════

const holistic = new Holistic({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
});

holistic.setOptions({
    modelComplexity: 1,         // Khớp với môi trường Python (mặc định là 1)
    smoothLandmarks: true,
    enableSegmentation: false,
    smoothSegmentation: false,
    refineFaceLandmarks: false, // Tắt refineFace — model Python chỉ lấy 468 điểm
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

// ── Xử lý mỗi frame ────────────────────────────────────────────────────────

holistic.onResults(async (results) => {
    // 1. Vẽ landmarks
    drawResults(results);

    // Bỏ qua nếu đang tạm dừng
    if (isPaused) return;

    // 2. Kiểm tra xem có tay trong khung hình không
    const hasHands = results.leftHandLandmarks || results.rightHandLandmarks;
    if (!hasHands) {
        noHandFrames++;
        if (noHandFrames > 15) {
            sequence = []; // Người dùng hạ tay -> Xóa sạch buffer
            predictionsList = []; // Xóa bộ nhớ đệm dự đoán
            frameBar.style.width   = '0%';
            frameCount.textContent = `0 / ${SEQ_LEN} frames`;
            latestKeypoints = null; // Ngừng thu thập lập tức
            if (detLabel.textContent !== '— Chờ nhận diện —' && !detLabel.textContent.includes('Đã tạm dừng')) {
                detLabel.textContent = '— Không phát hiện tay —';
            }
            return;
        }
    } else {
        noHandFrames = 0;
    }

    // Cập nhật tọa độ mới nhất (nhưng chưa đưa vào buffer ngay)
    latestKeypoints = extractKeypoints(results);
});

// ═══════════════════════════════════════════════════════════════════════════════
// ĐỒNG BỘ FPS (BÍ QUYẾT GIẢI QUYẾT LỆCH THỜI GIAN GIỮA WEB VÀ PYTHON)
// Chạy độc lập 33.3ms / lần (Đúng chuẩn 30 FPS của video Python).
// Nếu web lag (15 FPS), timer này sẽ tự động lặp lại frame cũ (Zero-Order Hold),
// giúp 30 frame của Web và Python đều có tổng thời gian múa dài ĐÚNG 1 GIÂY.
// ═══════════════════════════════════════════════════════════════════════════════
setInterval(() => {
    // Nếu đang tạm dừng, không có tay, hoặc đang chờ predict → bỏ qua
    if (isPaused || !latestKeypoints || predicting) return;

    // Đẩy frame vào buffer
    sequence.push(latestKeypoints);

    // Cập nhật progress bar
    const progress = sequence.length;
    frameBar.style.width   = `${(progress / SEQ_LEN) * 100}%`;
    frameCount.textContent = `${progress} / ${SEQ_LEN} frames`;

    // Gửi dự đoán khi đủ 30 frames
    if (sequence.length === SEQ_LEN) {
        predicting = true;
        const seqToSend = [...sequence];

        // Hiển thị trạng thái đang xử lý, giữ thanh ở 100%
        detLabel.textContent = '⏳ Đang nhận diện…';

        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ keypoints: seqToSend })
        })
        .then(res => res.json())
        .then(data => {
            if (data.top10) updateProbDisplay(data.top10);

            if (data.probability >= THRESHOLD) {
                detLabel.textContent = `✅ Nhận diện: ${data.action} (${(data.probability * 100).toFixed(1)}%)`;
                
                // Chỉ thêm từ mới nếu khác với từ cuối cùng đã nhận diện
                if (detectedWords.length === 0 || detectedWords[detectedWords.length - 1] !== data.action) {
                    detectedWords.push(data.action);
                    addChip(data.action);
                }
            } else {
                detLabel.textContent = `🔍 Chưa rõ (${(data.probability * 100).toFixed(1)}%)`;
            }
        })
        .catch(err => {
            console.error(err);
            detLabel.textContent = '❌ Lỗi nhận diện';
        })
        .finally(() => {
            // Theo yêu cầu: Ngay khi Model dự đoán xong, reset buffer và cho phép thu thập lượt mới ngay lập tức
            sequence = [];
            frameBar.style.width   = '0%';
            frameCount.textContent = `0 / ${SEQ_LEN} frames`;
            predicting = false;
        });
    }
}, 33); // 33.3ms ~ 30 FPS

// ═══════════════════════════════════════════════════════════════════════════════
// CAMERA — Khởi động webcam
// ═══════════════════════════════════════════════════════════════════════════════

statusText.textContent = 'Đang khởi động camera…';

// Trạng thái khóa để tránh dồn ứ frame cho MediaPipe
let isProcessing = false;

const camera = new Camera(video, {
    onFrame: async () => {
        // Nếu MediaPipe đang bận xử lý frame trước, ta bỏ qua frame này để webcam không bị đứng
        if (isProcessing) return;
        
        isProcessing = true;
        // KHÔNG dùng await ở đây để luồng video được tiếp tục trơn tru
        holistic.send({ image: video }).finally(() => {
            isProcessing = false;
        });
    },
    width: 480,
    height: 270
});

camera.start()
    .then(() => {
        canvas.width  = 480;
        canvas.height = 270;
        dot.classList.add('on');
        statusText.textContent = 'Đang hoạt động';
        console.log('✅ Camera & MediaPipe ready');
    })
    .catch((err) => {
        console.error('Camera error:', err);
        statusText.textContent = 'Lỗi camera!';
        detLabel.textContent = '❌ Không thể truy cập camera';
    });
