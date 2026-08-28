if env && {INTELLIGENCE_MODE}:
  curl -s {INTELLIGENCE_ENDPOINT}/policy >req.body만 추출> examples/support_platform/platform/config/policy.yaml

uvicorn support_platform.server:app --app-dir examples --port 8078