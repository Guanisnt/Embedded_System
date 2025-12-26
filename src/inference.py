import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import argparse
import time
import cv2
import mediapipe as mp


def allocate_buffers(engine, context):
    inputs = []
    outputs = []
    allocations = []

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dims = engine.get_tensor_shape(name)
        print(f"Tensor:{name}, Shape:{dims}")
        dtype = trt.nptype(engine.get_tensor_dtype(name))

        if dims[0] == -1:
            vol = 1
            for d in dims[1:]:
                vol *= d
            size = vol
        else:
            size = trt.volume(dims)

        host_mem = cuda.pagelocked_empty(size, dtype=dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        allocations.append(device_mem)

        dic = {
            "name": name,
            "host_mem": host_mem,
            "device_mem": device_mem,
            "shape": dims,
            "dtype": dtype
        }

        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(dic)
        else:
            outputs.append(dic)

    stream = cuda.Stream()
    return inputs, outputs, allocations, stream


def do_inference(context, frame, inputs, outputs, stream):
    np.copyto(inputs[0]["host_mem"], frame.ravel())

    for inp in inputs:
        cuda.memcpy_htod_async(inp['device_mem'], inp['host_mem'], stream)

    for inp in inputs:
        context.set_tensor_address(inp['name'], int(inp['device_mem']))
    for out in outputs:
        context.set_tensor_address(out['name'], int(out['device_mem']))

    context.execute_async_v3(stream_handle=stream.handle)

    for out in outputs:
        cuda.memcpy_dtoh_async(out["host_mem"], out["device_mem"], stream)

    stream.synchronize()
    return outputs[0]["host_mem"]


def load_engine(trt_runtime, plan_path):
    with open(plan_path, 'rb') as f:
        engine_data = f.read()
    return trt_runtime.deserialize_cuda_engine(engine_data)


def preprocess_frame(frame, width, height):
    """原本的預處理函數 - 完全保持不變"""
    frame_resized = cv2.resize(frame, (width, height))
    input_data = np.array(frame_resized, dtype=np.float32)
    mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    input_data -= mean
    input_data = np.ascontiguousarray(input_data).ravel()
    return input_data


class FaceDetector:
    """使用 MediaPipe 進行人臉偵測"""
    
    def __init__(self, min_detection_confidence=0.5, margin_ratio=0.3):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence
        )
        self.margin_ratio = margin_ratio
    
    def detect(self, frame):
        """偵測人臉並回傳臉部區域"""
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # 轉換為絕對座標
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)
                
                # 擴展邊界框
                margin_x = int(box_w * self.margin_ratio)
                margin_y = int(box_h * self.margin_ratio)
                
                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(w, x + box_w + margin_x)
                y2 = min(h, y + box_h + margin_y)
                
                # 裁剪臉部
                face_img = frame[y1:y2, x1:x2].copy()
                
                if face_img.size > 0:
                    faces.append({
                        'bbox': (x1, y1, x2 - x1, y2 - y1),
                        'face_img': face_img,
                        'confidence': detection.score[0]
                    })
        
        return faces
    
    def release(self):
        self.face_detection.close()


def main(opt):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)

    model_path = opt['engine']
    source = opt['source']
    HEIGHT, WIDTH = opt['imgsz'][0], opt['imgsz'][1]
    use_face_detect = opt['face_detect']
    threshold = opt['threshold']

    print(f"Loading Engine: {model_path}")
    print(f"Face Detection: {'ON' if use_face_detect else 'OFF'}")
    print(f"Threshold: {threshold}")
    
    try:
        engine = load_engine(trt_runtime, model_path)
    except FileNotFoundError:
        print(f"錯誤: 找不到引擎檔案 -> {model_path}")
        return

    context = engine.create_execution_context()
    inputs, outputs, allocations, stream = allocate_buffers(engine, context)

    # 初始化人臉偵測器（如果啟用）
    face_detector = None
    if use_face_detect:
        face_detector = FaceDetector(
            min_detection_confidence=opt['face_confidence'],
            margin_ratio=opt['face_margin']
        )
        print("MediaPipe Face Detection 初始化完成")

    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        print(f"Starting Webcam: {source}")
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit, 'f' to toggle face detection, 's' to save cropped face.")
    print("Press '+'/'-' to adjust face margin.")

    prev_frame_time = 0
    current_margin = opt['face_margin']
    show_cropped_face = True  # 顯示裁剪的臉部視窗

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.perf_counter()

        if use_face_detect and face_detector:
            # ===== 模式 1: 使用人臉偵測 =====
            faces = face_detector.detect(frame)
            
            for idx, face_info in enumerate(faces):
                face_img = face_info['face_img']
                x, y, w, h = face_info['bbox']
                
                # 使用和原本完全相同的預處理函數
                im = preprocess_frame(face_img, WIDTH, HEIGHT)
                
                # 推論
                out_raw = do_inference(context, im, inputs, outputs, stream)
                prob_value = float(out_raw[0])
                
                # 判斷結果
                if prob_value > threshold:
                    label = "Angry"
                    score = prob_value
                    color = (0, 0, 255)
                else:
                    label = "Normal"
                    score = 1 - prob_value
                    color = (0, 255, 0)
                
                # 繪製結果
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label_text = f"{label}: {score:.2f} (p={prob_value:.3f})"
                cv2.putText(frame, label_text, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # 顯示裁剪的臉部（只顯示第一張）
                if idx == 0 and show_cropped_face:
                    face_display = cv2.resize(face_img, (224, 224))
                    cv2.imshow('Cropped Face (compare with RAF-DB)', face_display)
            
            # 如果沒有偵測到人臉
            if len(faces) == 0:
                cv2.putText(frame, "No face detected", 
                           (frame.shape[1]//2 - 100, frame.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        else:
            # ===== 模式 2: 使用整張畫面（原本的方式）=====
            im = preprocess_frame(frame, WIDTH, HEIGHT)
            out_raw = do_inference(context, im, inputs, outputs, stream)
            prob_value = float(out_raw[0])
            
            if prob_value > threshold:
                label = "Angry"
                score = prob_value
                color = (0, 0, 255)
            else:
                label = "Normal"
                score = 1 - prob_value
                color = (0, 255, 0)
            
            info_text = f"{label}: {score:.2f}"
            cv2.putText(frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        end_time = time.perf_counter()
        infer_time = (end_time - start_time) * 1000

        # 計算 FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time

        # 顯示資訊
        mode_text = "Face Detect" if use_face_detect else "Full Frame"
        margin_text = f"margin={current_margin:.1f}" if use_face_detect else ""
        cv2.putText(frame, f"FPS: {int(fps)} | {mode_text} {margin_text} | {infer_time:.1f}ms", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow('Anger Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            use_face_detect = not use_face_detect
            if use_face_detect and face_detector is None:
                face_detector = FaceDetector(
                    min_detection_confidence=opt['face_confidence'],
                    margin_ratio=current_margin
                )
            print(f"Face Detection: {'ON' if use_face_detect else 'OFF'}")
        elif key == ord('s'):
            # 保存裁剪的臉部圖片
            if use_face_detect and face_detector:
                faces = face_detector.detect(frame)
                if len(faces) > 0:
                    timestamp = int(time.time())
                    filename = f"cropped_face_{timestamp}.jpg"
                    cv2.imwrite(filename, faces[0]['face_img'])
                    print(f"已保存裁剪的臉部: {filename}")
        elif key == ord('+') or key == ord('='):
            # 增加 margin
            current_margin = min(1.0, current_margin + 0.1)
            if face_detector:
                face_detector.margin_ratio = current_margin
            print(f"Face margin: {current_margin:.1f}")
        elif key == ord('-'):
            # 減少 margin
            current_margin = max(0.0, current_margin - 0.1)
            if face_detector:
                face_detector.margin_ratio = current_margin
            print(f"Face margin: {current_margin:.1f}")

    cap.release()
    if face_detector:
        face_detector.release()
    cv2.destroyAllWindows()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, required=True, help='Engine file path')
    parser.add_argument('--source', type=str, default='0', help='Webcam ID or video path')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[224, 224], help='inference size h,w')
    parser.add_argument('--threshold', type=float, default=0.4, help='Detection threshold')
    parser.add_argument('--face-detect', action='store_true', help='Enable face detection')
    parser.add_argument('--face-confidence', type=float, default=0.5, help='Face detection confidence')
    parser.add_argument('--face-margin', type=float, default=0.3, help='Face bbox margin ratio')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(vars(opt))
