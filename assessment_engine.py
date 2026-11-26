import os
import re
import shutil
import joblib
import numpy as np

# Add project root and venv\Scripts to PATH
project_dir = os.path.abspath(os.path.dirname(__file__))
venv_scripts = os.path.join(project_dir, "venv", "Scripts")
os.environ["PATH"] = project_dir + os.pathsep + venv_scripts + os.pathsep + os.environ.get("PATH", "")

# Configure bundled FFmpeg executable for Whisper ASR
try:
    import imageio_ffmpeg
    ffmpeg_src = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_src)
    ffmpeg_dst = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(ffmpeg_dst):
        shutil.copyfile(ffmpeg_src, ffmpeg_dst)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
except ImportError:
    pass

import whisper

LEVEL_LABELS = {
    0: "Beginner",
    1: "Developing",
    2: "Proficient",
    3: "Advanced"
}

# Load Whisper ASR model once at startup ("base" balances speed and accuracy for English & Filipino)
whisper_model = whisper.load_model("base")

# Load trained Random Forest classifier
RF_MODEL_PATH = os.path.join("models", "rf_reading_model.pkl")
rf_classifier = joblib.load(RF_MODEL_PATH)


def clean_and_tokenize(text: str) -> list[str]:
    """Normalizes casing and removes punctuation for fair word-level comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", "", text)
    return [word for word in text.split() if word]


def transcribe_audio(audio_file_path: str, language: str = "en") -> str:
    """
    Converts student voice recording to text using OpenAI Whisper ASR.
    language: 'en' for English, 'tl' for Filipino/Tagalog
    """
    lang_code = "tl" if language.lower() in ["filipino", "tagalog", "tl"] else "en"
    result = whisper_model.transcribe(audio_file_path, language=lang_code, fp16=False)
    return result.get("text", "").strip()


def compute_word_levenshtein(reference_text: str, hypothesis_text: str) -> dict:
    """
    Computes word-level Levenshtein Distance, Substitutions (S), Deletions (D),
    Insertions (I), Word Error Rate (WER), Reading Accuracy (%), and word-by-word error list.
    """
    ref_words = clean_and_tokenize(reference_text)
    hyp_words = clean_and_tokenize(hypothesis_text)

    n = len(ref_words)
    m = len(hyp_words)

    if n == 0:
        return {
            "total_words": 0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": m,
            "wer": 100.0 if m > 0 else 0.0,
            "reading_accuracy": 0.0,
            "error_list": []
        }

    # Dynamic programming matrix for edit distance
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # Deletion
                dp[i][j - 1] + 1,        # Insertion
                dp[i - 1][j - 1] + cost  # Match or Substitution
            )

    # Backtrack to identify exact S, D, I operations and build Error List
    i, j = n, m
    substitutions = 0
    deletions = 0
    insertions = 0
    error_list = []

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            error_list.append({
                "position": i,
                "original": ref_words[i - 1],
                "read_as": hyp_words[j - 1],
                "type": "SUB"
            })
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            error_list.append({
                "position": i,
                "original": ref_words[i - 1],
                "read_as": "(skipped)",
                "type": "DEL"
            })
            i -= 1
        else:
            insertions += 1
            error_list.append({
                "position": i + 1,
                "original": "(none)",
                "read_as": f"+ {hyp_words[j - 1]}",
                "type": "INS"
            })
            j -= 1

    error_list.reverse()

    total_errors = substitutions + deletions + insertions
    wer = round((total_errors / n) * 100.0, 2)
    reading_accuracy = round(max(0.0, (1.0 - (total_errors / n)) * 100.0), 2)

    return {
        "total_words": n,
        "words_read": m,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "wer": wer,
        "reading_accuracy": reading_accuracy,
        "error_list": error_list
    }


def compute_wpm(words_read: int, duration_seconds: float) -> float:
    """Computes Words Per Minute (WPM) fluency metric."""
    if duration_seconds <= 0:
        return 0.0
    minutes = duration_seconds / 60.0
    return round(words_read / minutes, 2)


def evaluate_comprehension(student_answers: list[str], correct_answers: list[str]) -> dict:
    """
    Evaluates multiple-choice comprehension answers and computes semantic token overlap
    for NLP-supported answer verification.
    """
    if not correct_answers:
        return {"correct_count": 0, "total_questions": 0, "comprehension_score": 0.0}

    total_q = len(correct_answers)
    correct_count = 0

    for idx, expected in enumerate(correct_answers):
        if idx >= len(student_answers):
            continue
        ans = student_answers[idx].strip().lower()
        exp = expected.strip().lower()

        if ans == exp:
            correct_count += 1
        else:
            # NLP token-level Jaccard similarity fallback for text answers
            ans_tokens = set(clean_and_tokenize(ans))
            exp_tokens = set(clean_and_tokenize(exp))
            if exp_tokens and (len(ans_tokens & exp_tokens) / len(exp_tokens)) >= 0.8:
                correct_count += 1

    score = round((correct_count / total_q) * 100.0, 2)
    return {
        "correct_count": correct_count,
        "total_questions": total_q,
        "comprehension_score": score
    }


def classify_reading_level(reading_accuracy: float, wpm: float, comprehension_score: float) -> dict:
    """Uses the trained Random Forest classifier to predict the student's reading level."""
    features = np.array([[reading_accuracy, wpm, comprehension_score]])
    pred_class = int(rf_classifier.predict(features)[0])
    probabilities = rf_classifier.predict_proba(features)[0]

    return {
        "reading_level": LEVEL_LABELS[pred_class],
        "confidence": round(float(np.max(probabilities)) * 100.0, 2)
    }