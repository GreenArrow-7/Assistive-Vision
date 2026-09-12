"""Central configuration for the Assistive Vision server."""

# ---------- Models ----------
OBJECT_MODEL = "yolov8n.pt"          # COCO-pretrained, auto-downloads on first run
OBJECT_MODEL_CUSTOM = "models/av_obstacle.pt"  # your fine-tuned model (auto-used if present)
TEXT_OBB_MODEL = "models/text_obb.pt"  # optional fine-tuned YOLO-OBB text model
OCR_LANGS = ["en"]
OCR_GPU = False                       # set True if CUDA available

# ---------- Detection thresholds ----------
OBJ_CONF = 0.45
TEXT_CONF = 0.40                      # EasyOCR confidence 0-1
MAX_ANNOUNCE = 6                      # cap items per spoken summary
OCR_EVERY_N = 3                       # live mode: EasyOCR dominates CPU cost;
                                      # run it every Nth frame, reuse in between
CONFIRM_IOU = 0.2                     # live frames are 2.6 s apart while walking,
                                      # so boxes shift a lot; keep the overlap gate loose

# ---------- Spatial ----------
NEAR_AREA_RATIO = 0.06                # bbox area / frame area => "very close"
MID_AREA_RATIO = 0.015                # => "nearby", else "at a distance"

# COCO classes considered HAZARDS (announced first, hazard tone)
HAZARD_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train",
    "dog", "fire hydrant", "stop sign", "chair", "bench", "couch",
    "potted plant", "dining table", "traffic light",
}

# COCO classes worth announcing as everyday objects (non-hazard)
OBJECT_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "backpack",
    "umbrella", "handbag", "suitcase", "bottle", "cup", "chair", "couch",
    "bed", "dining table", "toilet", "tv", "laptop", "cell phone", "book",
    "door", "bench", "potted plant", "sink", "refrigerator", "stop sign",
    "traffic light", "fire hydrant", "dog", "cat", "train",
}

# ---------- Symbol recognition ----------
# Symbols resolved two ways:
#  1) COCO class → symbol (e.g. 'toilet' object implies washroom area)
#  2) OCR keyword → symbol (signboards usually carry these words/icons+text)
SYMBOL_FROM_CLASS = {
    "toilet": "washroom",
    "stop sign": "stop sign",
    "traffic light": "traffic signal",
}
SYMBOL_KEYWORDS = {
    "washroom":        ["washroom", "restroom", "toilet", "wc", "gents", "ladies", "men", "women"],
    "emergency exit":  ["emergency exit", "fire exit"],
    "exit": ["exit", "way out"],
    "elevator":        ["lift", "elevator"],
    "wheelchair access": ["wheelchair", "accessible", "disabled"],
    "parking":         ["parking", "car park", "p1", "p2", "basement parking"],
    "cafeteria":       ["cafeteria", "canteen", "cafe", "food court", "restaurant"],
    "no smoking":      ["no smoking", "smoking prohibited"],
    "reception":       ["reception", "front desk", "help desk", "information", "enquiry"],
    "pharmacy":        ["pharmacy", "chemist", "medical", "dispensary"],
    "stairs":          ["stairs", "staircase", "steps"],
}

# words stripped from voice queries: "where is the exit" -> "exit"
QUERY_STOPWORDS = {
    "where", "is", "the", "a", "an", "find", "locate", "search", "for",
    "me", "show", "navigate", "to", "please", "nearest", "go", "take",
}

# ---------- Step-distance estimation (monocular pinhole model) ----------
# distance Z = f_px * H_real / h_box_px ;  f_px = (frame_h/2) / tan(VFOV/2)
CAMERA_VFOV_DEG = 50.0        # typical phone rear camera vertical FOV
STEP_LENGTH_M = 0.75          # average adult step
MAX_STEPS_ANNOUNCE = 30       # beyond this say "far away"

# Known real-world heights (meters) per class — enables metric distance
KNOWN_HEIGHTS = {
    "person": 1.70, "bicycle": 1.00, "car": 1.50, "motorcycle": 1.20,
    "bus": 3.00, "truck": 3.20, "train": 3.50, "dog": 0.50, "cat": 0.30,
    "chair": 0.90, "couch": 0.85, "bench": 0.85, "bed": 0.60,
    "dining table": 0.75, "toilet": 0.75, "potted plant": 0.60,
    "tv": 0.60, "laptop": 0.25, "bottle": 0.25, "cup": 0.12,
    "backpack": 0.50, "handbag": 0.30, "suitcase": 0.60, "umbrella": 0.80,
    "refrigerator": 1.70, "sink": 0.85, "book": 0.24, "cell phone": 0.15,
    "stop sign": 0.75, "traffic light": 0.90, "fire hydrant": 0.75,
}

# ---------- Accuracy hardening ----------
CAMERA_HEIGHT_M = 1.45        # phone held at chest height (ground-plane model)
BLUR_THRESHOLD = 55.0         # variance-of-Laplacian below this = too blurry
EDGE_MARGIN_PX = 6            # bbox within this of frame edge = truncated
FEET_AREA_RATIO = 0.15        # bottom-touching + huge box => object at feet

# Per-class confidence overrides (noisy classes need stricter gates)
CLASS_CONF = {
    "person": 0.62, "bed": 0.60, "couch": 0.58, "dog": 0.55, "cat": 0.55,
    "handbag": 0.55, "backpack": 0.55, "tv": 0.55,
}


# Environment overrides are read at process startup. Uvicorn --env-file loads .env.
import os

def _number(name, default, low, high, integer=False):
    value = float(os.getenv(name, str(default)))
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value) if integer else value

OBJECT_MODEL = os.getenv("OBJECT_MODEL_PATH", OBJECT_MODEL)
OBJECT_MODEL_CUSTOM = os.getenv("CUSTOM_OBJECT_MODEL_PATH", OBJECT_MODEL_CUSTOM)
TEXT_OBB_MODEL = os.getenv("TEXT_MODEL_PATH", TEXT_OBB_MODEL)
OCR_LANGS = os.getenv("OCR_LANGUAGES", "en").split(",")
OCR_GPU = os.getenv("OCR_GPU", "false").lower() == "true"
DEVICE = os.getenv("DEVICE", "cpu")
OBJ_CONF = _number("CONFIDENCE_THRESHOLD", OBJ_CONF, .01, 1)
TEXT_CONF = _number("TEXT_CONFIDENCE_THRESHOLD", TEXT_CONF, .01, 1)
IMAGE_SIZE = _number("IMAGE_SIZE", 640, 320, 1280, True)
NMS_THRESHOLD = _number("NMS_THRESHOLD", .5, .05, .95)
OCR_EVERY_N = _number("OCR_INTERVAL", OCR_EVERY_N, 1, 30, True)
FRAME_INTERVAL = _number("FRAME_INTERVAL", 350, 100, 30000, True)
ANNOUNCEMENT_COOLDOWN = _number("ANNOUNCEMENT_COOLDOWN", 9, 1, 120)
STEP_LENGTH_M = _number("STEP_LENGTH", STEP_LENGTH_M, .2, 1.5)
CAMERA_HEIGHT_M = _number("CAMERA_HEIGHT", CAMERA_HEIGHT_M, .3, 2.5)
CAMERA_VFOV_DEG = _number("CAMERA_VFOV", CAMERA_VFOV_DEG, 25, 90)
MAX_ANNOUNCE = _number("MAX_ANNOUNCE", 3, 1, 6, True)
TTS_RATE = _number("TTS_RATE", 1.05, .5, 2)
LANGUAGE = os.getenv("LANGUAGE", "en-IN")
# Only these classes can plausibly supply a floor contact point.
GROUND_CLASSES = {"person", "chair", "table", "dining table", "door", "dustbin",
                  "car", "bus", "truck", "bicycle", "motorcycle", "bench", "couch",
                  "sink", "toilet", "refrigerator", "pole", "potted plant"}

# Live lanes run independently; these values are milliseconds.
OCR_INTERVAL_MS = _number("OCR_INTERVAL_MS", 2000, 500, 30000, True)
TTS_COOLDOWN_MS = _number("TTS_COOLDOWN_MS", 650, 200, 3000, True)
DETECTION_TTL_MS = _number("DETECTION_TTL_MS", 6500, 1000, 30000, True)
MAX_MISSED_FRAMES = _number("MAX_MISSED_FRAMES", 3, 1, 10, True)
MAX_RESULT_AGE_MS = _number("MAX_RESULT_AGE_MS", 4500, 1000, 15000, True)
LOCATION_TIMEOUT_MS = _number("LOCATION_TIMEOUT_MS", 8000, 1000, 15000, True)
INFERENCE_THREADS = _number("INFERENCE_THREADS", 2, 1, 8, True)
