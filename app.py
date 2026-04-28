"""
Sign Language Recognition - Flask Backend
==========================================
Sử dụng model LSTM (best_model.keras) để nhận diện cử chỉ ngôn ngữ ký hiệu
từ keypoints gửi từ trình duyệt, sau đó dùng Gemini AI để sinh câu tiếng Việt.
"""

import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ─── Load biến môi trường từ file .env ──────────────────────────────────────
load_dotenv()

# ─── Gemini AI (Sử dụng SDK chính thức) ─────────────────────────────────────
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

app = Flask(__name__)
CORS(app)

# ─── Load model & normalization params ─────────────────────────────
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "best_model.keras")
MEAN_PATH    = os.path.join(os.path.dirname(__file__), "norm_mean.npy")
STD_PATH     = os.path.join(os.path.dirname(__file__), "norm_std.npy")

model     = tf.keras.models.load_model(MODEL_PATH)
norm_mean = np.load(MEAN_PATH)
norm_std  = np.load(STD_PATH)

# tránh chia cho 0
norm_std = np.where(norm_std == 0, 1e-6, norm_std)

with open("./actions.json", "r", encoding="utf-8") as f:
    actions = json.load(f)

print("Loaded actions:", actions)
print("Model input shape:", model.input_shape)

# ─── Gemini setup ─────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING — sao chép chính xác từ notebook Model_Selected_Final.ipynb
# ═══════════════════════════════════════════════════════════════════════════════

# ── Face landmark indices (chọn lọc) ────────────────────────────────────────
IDX_VIEN_MAT = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]
IDX_MAT_TRAI = [33,160,158,133,153,144,145,163,7,246,226,110,24,23,22,26,112,243,190,56,28,27,29,30,247]
IDX_MAT_PHAI = [362,385,387,263,373,380,374,390,249,466,463,341,256,252,253,254,339,446,467,260,259,257,258,286,414]
IDX_LONG_MAY_TRAI = [70,63,105,66,107,55,65,52,53,46]
IDX_LONG_MAY_PHAI = [336,296,334,293,300,276,283,282,295,285]
IDX_MIENG = [
    61,146,91,181,84,17,314,405,
    321,375,291,409,270,269,267,0,37,39,40,185,
    308,324,318,402,317,14,87,178,88,95,78,191,
    80,81,82,13,312,311,310,415
]

SELECTED_FACE_IDX = sorted(set(
    IDX_VIEN_MAT +
    IDX_MAT_TRAI + IDX_MAT_PHAI +
    IDX_LONG_MAY_TRAI + IDX_LONG_MAY_PHAI +
    IDX_MIENG
))

# Pre-compute face feature indices (trong mảng 1662 gốc)
FACE_FEATURE_INDICES = []
for idx in SELECTED_FACE_IDX:
    start = 132 + idx * 3
    FACE_FEATURE_INDICES.extend([start, start + 1, start + 2])
FACE_FEATURE_INDICES = np.array(FACE_FEATURE_INDICES)

print(f"📐 Selected face landmarks: {len(SELECTED_FACE_IDX)} điểm → {len(FACE_FEATURE_INDICES)} features")


def extract_selected_keypoints(X):
    """
    Trích xuất keypoints đã chọn từ bộ 1662 feature gốc.
    Input:  X shape (30, 1662)
    Output: X_new shape (30, 696) = pose(132) + face_selected(438) + lh(63) + rh(63)
    """
    pose = X[:, 0:132]
    lh   = X[:, 1536:1599]
    rh   = X[:, 1599:1662]
    face = X[:, FACE_FEATURE_INDICES]
    X_new = np.concatenate([pose, face, lh, rh], axis=1)
    return X_new.astype(np.float32)


def standardize_keypoints(X_data):
    """
    Chuẩn hóa keypoints — SAO CHÉP CHÍNH XÁC từ notebook Model_selected_final.ipynb
    Pipeline: Forward-fill → Backward-fill → EMA → Nose fallback → Centering
    (KHÔNG có bước Scaling — notebook không dùng)
    Input:  X_data shape (30, 696) hoặc (N, 30, 696)
    Output: X_cent cùng shape
    """
    X_cent = X_data.copy()
    is_2d = False
    if len(X_cent.shape) == 2:
        is_2d = True
        X_cent = np.expand_dims(X_cent, axis=0)  # (1, 30, 696)

    # BƯỚC 1a: Nội suy TIẾN (Forward-fill) — chống mất dấu
    for b in range(X_cent.shape[0]):
        for t in range(1, X_cent.shape[1]):
            zeros_mask = (X_cent[b, t, :] == 0.0)
            X_cent[b, t, zeros_mask] = X_cent[b, t - 1, zeros_mask]

    # BƯỚC 1b: Nội suy LÙI (Backward-fill) — xử lý nốt nếu frame 0 bị lỗi
    for b in range(X_cent.shape[0]):
        for t in range(X_cent.shape[1] - 2, -1, -1):
            zeros_mask = (X_cent[b, t, :] == 0.0)
            X_cent[b, t, zeros_mask] = X_cent[b, t + 1, zeros_mask]

    # BƯỚC 2: EMA chống rung
    alpha = 0.6
    for b in range(X_cent.shape[0]):
        for t in range(1, X_cent.shape[1]):
            X_cent[b, t, :] = alpha * X_cent[b, t, :] + (1 - alpha) * X_cent[b, t - 1, :]

    # BƯỚC 3: Centering an toàn (Neo điểm Mũi)
    nose_x = X_cent[:, :, 0:1].copy()
    nose_y = X_cent[:, :, 1:2].copy()
    nose_z = X_cent[:, :, 2:3].copy()

    # Dự phòng: Nếu mũi Pose bị 0, dùng mũi của Face (index 132)
    mask_zero = (nose_x == 0)
    nose_x[mask_zero] = X_cent[:, :, 132:133][mask_zero]
    nose_y[mask_zero] = X_cent[:, :, 133:134][mask_zero]
    nose_z[mask_zero] = X_cent[:, :, 134:135][mask_zero]

    # Centering — pose (stride=4: x,y,z,visibility)
    X_cent[:, :, 0:132:4] -= nose_x
    X_cent[:, :, 1:132:4] -= nose_y
    X_cent[:, :, 2:132:4] -= nose_z

    # Centering — face + hands (stride=3: x,y,z)
    X_cent[:, :, 132::3] -= nose_x
    X_cent[:, :, 133::3] -= nose_y
    X_cent[:, :, 134::3] -= nose_z

    # (Notebook KHÔNG dùng Scaling khi train → inference cũng KHÔNG dùng)

    if is_2d:
        X_cent = X_cent.squeeze(axis=0)

    return X_cent


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Trang chính — render giao diện webcam."""
    return render_template('index.html', actions=json.dumps(actions, ensure_ascii=False))


@app.route('/predict', methods=['POST'])
def predict():
    """
    Nhận 30 frames × 1662 keypoints từ client,
    chạy pipeline tiền xử lý + model prediction,
    trả về top-5 kết quả.
    """
    try:
        data = request.get_json()
        keypoints = np.array(data['keypoints'], dtype=np.float32)  # (30, 1662)

        # Pipeline tiền xử lý — đúng thứ tự như notebook
        # Bước 1: Chọn lọc keypoints (1662 → 696)
        keypoints = extract_selected_keypoints(keypoints)  # (30, 696)

        # Bước 2: Chuẩn hóa (centering + scaling)
        keypoints = standardize_keypoints(keypoints)  # keypoints shape (30, 696)

        # Bước 3: Z-score normalization
        zero_mask = (keypoints == 0.0)
        keypoints = (keypoints - norm_mean) / (norm_std + 1e-8)
        keypoints[zero_mask] = 0.0

        # Bước 4: Thêm batch dimension + predict
        keypoints = np.expand_dims(keypoints, axis=0)      # (1, 30, 696)
        prediction = model.predict(keypoints, verbose=0)[0]  # (50,)

        # Top 10 kết quả (theo yêu cầu 5-10 cử chỉ)
        top_indices = np.argsort(prediction)[::-1][:10]
        top10 = []
        for idx in top_indices:
            top10.append({
                'action': actions[int(idx)],
                'probability': float(prediction[idx])
            })

        best_action = top10[0]['action']
        best_prob   = top10[0]['probability']

        return jsonify({
            'action': best_action,
            'probability': best_prob,
            'top10': top10
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/generate', methods=['POST'])
def generate():
    """
    Nhận danh sách từ đã nhận diện, gửi cho Gemini AI
    để sinh câu tiếng Việt tự nhiên nhất.
    Có cơ chế retry với exponential backoff khi gặp lỗi 429 (rate limit).
    """
    import time

    data = request.get_json()
    words = data.get('words', [])

    if not words:
        return jsonify({'sentence': ''})

    prompt = (
        "Bạn là trợ lý dịch ngôn ngữ ký hiệu Việt Nam. "
        "Tôi đưa cho bạn danh sách các từ/cụm từ tiếng Việt (không dấu) "
        "được nhận diện từ camera theo thứ tự thời gian:\n\n"
        f"[{', '.join(words)}]\n\n"
        "Hãy ghép các từ này thành MỘT câu tiếng Việt CÓ DẤU, "
        "tự nhiên, đúng ngữ pháp và có nghĩa nhất có thể. "
        "Nếu có thể, hãy thêm giới từ, chủ ngữ hoặc liên từ cho câu mạch lạc.\n"
        "Chỉ trả về câu kết quả duy nhất, không giải thích, không thêm dấu ngoặc kép."
    )

    # Retry với exponential backoff khi bị rate limit (429)
    max_retries = 3
    base_delay = 2  # giây

    for attempt in range(max_retries + 1):
        try:
            response = gemini_model.generate_content(prompt)
            sentence = response.text.strip()
            return jsonify({'sentence': sentence})

        except google_exceptions.ResourceExhausted as e:
            if attempt < max_retries:
                wait_time = base_delay * (2 ** attempt)  # 2s, 4s, 8s
                print(f"⏳ Gemini 429 rate limit — chờ {wait_time}s rồi thử lại (lần {attempt + 1}/{max_retries})…")
                time.sleep(wait_time)
                continue
            else:
                print("❌ Gemini 429 — đã hết lần retry")
                fallback_sentence = " ".join(words)
                return jsonify({
                    'sentence': f'{fallback_sentence} (AI đang quá tải)'
                }), 200

        except google_exceptions.GoogleAPIError as e:
            print(f"❌ Gemini API error: {e}")
            fallback_sentence = " ".join(words)
            return jsonify({'sentence': f'{fallback_sentence} (Lỗi API)'}), 200
            
        except Exception as e:
            print(f"❌ Gemini error: {e}")
            fallback_sentence = " ".join(words)
            return jsonify({'sentence': f'{fallback_sentence} (Lỗi AI)'}), 200

    fallback_sentence = " ".join(words)
    return jsonify({'sentence': f'{fallback_sentence} (Không thể kết nối AI)'}), 200


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n🚀 Server đang chạy tại http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
