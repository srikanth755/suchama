import cv2
from ultralytics import YOLO
from ultralytics import RTDETR

# Load the trained model
# model = YOLO('/Users/saisrikanth/Desktop/suchama assignment/best (3).pt')
# model = RTDETR("/Users/saisrikanth/Desktop/suchama assignment/best (3).pt")
# Input/output paths
# input_video_path = '/Users/saisrikanth/Desktop/suchama assignment/data/Operation_1.0.mov'
# output_video_path = '/Users/saisrikanth/Desktop/suchama assignment/inference/Operation_1.0.3.mov'

def get_class_conf(result, T):
    boxes = result[0].boxes  # get Boxes object

    ids = boxes.cls.cpu().numpy().astype(int).tolist()  # tensor of class indices
    scores = boxes.conf.cpu().numpy().tolist()  # tensor of confidence scores
    # Optional: map class_ids to class names if available
    class_names = result[0].names
    if ids and scores and len(ids) == len(scores):
        best_id, best_score = max(zip(ids, scores), key=lambda x: x[1])
        return (class_names[best_id], best_score, T)
    else:
        return ()

def yolo_inference(input_video_path, output_video_path):

    # Load the trained model
    model = YOLO('/Users/saisrikanth/Desktop/suchama assignment/best (4).pt')
    # model = RTDETR("/Users/saisrikanth/Desktop/suchama assignment/best (3).pt")
    print(f'Loaded the model')

    # Read the input video
    cap = cv2.VideoCapture(input_video_path)
    print(f'reading the file completed')

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    res = []

    print(f'Starting')
    # Process frame by frame
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame is None or frame.size == 0:
            print(f"[Warning] Reached end of video or invalid frame at {frame_count}")
            break

        # Run detection
        results = model.predict(frame, verbose=False)
        T = frame_count/fps

        pred = get_class_conf(results, T)

        if len(pred)>0:
            res.append(pred)

        # print('Got results')

        # Annotate frame
        annotated_frame = results[0].plot()
        # print('output frame')

        # Write annotated frame to output video
        out.write(annotated_frame)
        # print('Written the output')

        frame_count += 1
        # print(f"Processed frame {frame_count}")
        del frame
        del annotated_frame
        del results

        # Cleanup
    cap.release()
    out.release()
    return output_video_path, res

def yolo_inference1(input_video_path, output_video_path):
    # Load the trained model
    model = YOLO('/Users/saisrikanth/Desktop/suchama assignment/best (4).pt')
    print('Loaded the model')

    # Read the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("[Error] Failed to open video file.")
        return None, []

    print('Reading the file completed')

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    res = []
    print('Starting inference...')

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success or frame is None or frame.size == 0:
            print(f"[Info] Done processing video or encountered invalid frame at {frame_count}")
            break

        # Run detection
        results = model.predict(frame, verbose=False)
        T = frame_count / fps
        pred = get_class_conf(results, T)

        if pred:
            res.append(pred)

        # Annotate and write the frame
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        # Immediate memory management hint (optional, helps with big videos)
        del frame
        del annotated_frame
        del results

        frame_count += 1

    cap.release()
    out.release()
    print(f"Inference completed. Processed {frame_count} frames.")
    return output_video_path, res
