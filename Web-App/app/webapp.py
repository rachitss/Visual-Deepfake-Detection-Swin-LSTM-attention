"""Streamlit front-end for the Deepfake Detection FastAPI service."""
import io
import os
from datetime import datetime
from typing import Tuple

import requests
import streamlit as st

# streamlit run app/webapp.py

DEFAULT_API_URL = os.getenv("DEEPFAKE_API_URL", "http://localhost:8000")
INFERENCE_ROUTE = "/inference"
HEALTH_ROUTE = "/health"
MAX_UPLOAD_MB = 200


def normalize_base_url(url: str) -> str:
	url = url.strip()
	if not url:
		return DEFAULT_API_URL
	return url.rstrip("/")


def check_health(api_base: str, timeout: int) -> Tuple[bool, str]:
	try:
		resp = requests.get(f"{api_base}{HEALTH_ROUTE}", timeout=timeout)
		resp.raise_for_status()
		payload = resp.json()
		device = payload.get("device", "unknown")
		return True, f"online ({device})"
	except requests.RequestException as exc:
		return False, str(exc)


def run_inference(api_base: str, file_buffer: io.BytesIO, filename: str, timeout: int):
	files = {
		"file": (
			filename,
			file_buffer.getvalue(),
			"application/octet-stream",
		)
	}
	resp = requests.post(f"{api_base}{INFERENCE_ROUTE}", files=files, timeout=timeout)
	resp.raise_for_status()
	return resp.json()


def main():
	st.set_page_config(page_title="Deepfake Detection", layout="centered")
	st.title("Deepfake Detection Client")
	st.write("Upload a video file and forward it to the FastAPI inference service.")

	status_box = st.empty()
	api_base = normalize_base_url(DEFAULT_API_URL)
	if "api_status" not in st.session_state:
		with st.spinner("Connecting to API..."):
			st.session_state.api_status = check_health(api_base, timeout=5)
	if st.button("Retry connection", type="secondary"):
		with st.spinner("Retrying connection..."):
			st.session_state.api_status = check_health(api_base, timeout=5)
	health_ok, health_message = st.session_state.api_status
	if health_ok:
		status_box.success(f"API {health_message}")
	else:
		status_box.warning(f"API unreachable: {health_message}")

	timeout = 600

	uploader = st.file_uploader(
		"Video file",
		type=["mp4"],
		help="Files over roughly %d MB can slow uploads." % MAX_UPLOAD_MB,
	)

	if uploader is not None:
		st.video(uploader)

	disabled = uploader is None
	if st.button("Run inference", type="primary", disabled=disabled):
		if uploader is None:
			st.error("Please upload a video first.")
		else:
			try:
				with st.spinner("Uploading video and awaiting prediction..."):
					result = run_inference(api_base, uploader, uploader.name, timeout)
			except requests.RequestException as exc:
				st.error(f"Request failed: {exc}")
			else:
				st.success("Inference completed.")
				prediction = result.get("prediction")
				probability = result.get("probability", 0.0)
				label = "Fake" if prediction == 1 else "Real"
				confidence = probability if prediction == 1 else 1 - probability
				st.markdown(
					f"### Prediction: {label}\n" f"### Confidence: {confidence:.3%}", 
					help=None,
				)
				st.caption(f"Last updated at {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
	main()

	