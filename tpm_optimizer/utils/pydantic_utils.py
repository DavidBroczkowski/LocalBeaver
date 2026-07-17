import json
import sys
import urllib.error
import urllib.request

class Agent:
    def __init__(
        self,
        model: str,
        system: str ,
        max_tokens: int,
        base_url: str,
        api_key: str | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ):
        self.model = model
        self.system = system
        self.max_tokens= max_tokens
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.total_tokens = 0

    def read_prompt(self, parts: list[str]) -> str:
        if parts:
            return " ".join(parts).strip()

        if not sys.stdin.isatty():
            return sys.stdin.read().strip()

        return input("Text to complete: ").strip()


    def make_request_payload(self, prompt: str) -> dict:
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
        }


        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_completion_tokens"] = self.max_tokens

        return payload


    def extract_completion(self, response_data: dict) -> str:
        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return json.dumps(response_data, indent=2, ensure_ascii=False)

        if isinstance(content, str):
            return content

        return json.dumps(content, indent=2, ensure_ascii=False)

    def _extract_token_usage(self, response_data: dict) -> None:
        try:
            self.completion_tokens += response_data["usage"]["completion_tokens"]
            self.prompt_tokens += response_data["usage"]["prompt_tokens"]
            self.total_tokens += response_data["usage"]["total_tokens"]
        except(KeyError, IndexError, TypeError) as e:
            import traceback
            print(f"[ERROR] Token Usage Extraction failed due to a formatting error: {e}")
            traceback.print_exc()



    def format_models(self, response_data: dict) -> str:
        models = response_data.get("data")
        if not isinstance(models, list):
            return json.dumps(response_data, indent=2, ensure_ascii=False)

        model_ids = [
            model.get("id")
            for model in models
            if isinstance(model, dict) and isinstance(model.get("id"), str)
        ]
        if not model_ids:
            return json.dumps(response_data, indent=2, ensure_ascii=False)

        return "\n".join(sorted(model_ids))


    def get_available_models(self) -> dict:
        endpoint = self.base_url.rstrip("/") + "/v1/models"
        request = urllib.request.Request(
            endpoint,
            method="GET",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API returned HTTP {error.code}: {details}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Could not reach API: {error.reason}") from error


    def post_chat_completion(
        self, payload: dict
    ) -> dict:
        endpoint = self.base_url.rstrip("/") + "/v1/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_data = json.loads(response.read().decode("utf-8"))
                self._extract_token_usage(response_data)
                return response_data
        except urllib.error.HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API returned HTTP {error.code}: {details}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Could not reach API: {error.reason}") from error
