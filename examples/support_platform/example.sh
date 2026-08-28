if env && {INTELLIGENCE_MODE}:
  curl -s {INTELLIGENCE_ENDPOINT}/policy >req.body만 추출> common/config/policy.yaml

cd examples/support_platform && uvicorn server:app --port 8078