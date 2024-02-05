import openai
from transformers import GPT2TokenizerFast


MODEL_MAXLEN = {
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-35-turbo": 4096,
    "gpt-35-turbo-0301": 4096,
}


class ChatGPT:
    """ A very simple wrapper around OpenAI's ChatGPT API. Makes it easy to create custom messages & chat. """
    
    def __init__(self, model="gpt-3.5-turbo", completion_hparams=None, api_key=None):
        if api_key is not None:
            openai.api_key = api_key
        self.model = model
        self.completion_hparams = completion_hparams or {}
        self.history = []
        self._messages = []
        self._system = "You are a helpful assistant."
        self._tok = GPT2TokenizerFast.from_pretrained("gpt2")
        self._tok.model_max_length = 10000000000000 # hack to avoid model maxlen warnings

    @property
    def messages(self):
        """ The messages object for the current conversation. """
        messages = [{"role": "system", "content": self._system}] + self._messages
        return messages

    @property
    def model_maxlen(self):
        """ The number of tokens the loaded model can fit in its context window. """
        return MODEL_MAXLEN[self.model]

    @property
    def client(self):
        return openai.OpenAI()

    def get_token_count(self, text):
        """ Returns the number of GPT tokens used in the provided text. """
        return len(self._tok.encode(text))

    def trim(self, text, token_length=None):
        """ Truncates text to token_length (default: the max length of self.model) """
        if token_length is None:
            token_length = self.model_maxlen - 500 # leave some wiggle room
        tokens = self._tok.encode(text)
        return self._tok.decode(tokens[:token_length])

    def system(self, message, do_reset=True):
        """ Set the system message and optionally reset the conversation (default=true) """
        if do_reset:
            self.reset()
        self._system = message

    def user(self, message):
        """ Add a user message to the conversation """
        self._messages.append({"role": "user", "content": message})

    def assistant(self, message):
        """ Add an assistant message to the conversation """
        self._messages.append({"role": "assistant", "content": message})

    def reset(self):
        """ Reset the conversation (does not reset the system message) """
        self._messages = []

    def _make_completion(self, messages, **completion_hparams):
        """ Makes a completion with the current messages """
        kwargs = self.completion_hparams.copy()
        kwargs.update(completion_hparams)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )

        self.history.append((messages, completion))
        return completion

    def _make_completion_stream(self, messages=None, like_api=False, **completion_hparams):
        """ Makes a streaming completion with the current messages """
        kwargs = self.completion_hparams.copy()
        kwargs.update(completion_hparams)
        if messages is None:
            messages = self.messages

        completion_gen = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            **kwargs
        )

        chunk_contents = []
        for chunk in completion_gen:
            print(chunk.model_dump_json())
            content = chunk.choices[0].delta.content
            chunk_contents.append(content)
            if chunk.choices[0].finish_reason is not None:
                return

            if like_api:
                yield chunk
            else:
                yield content

        self.history.append((messages, "".join(chunk_contents)))


    def call(self, stream=False):
        """ Call ChatGPT with the current messages and return the assitant's message """
        if stream:
            return self._make_completion_stream(self.messages)
        else:
            completion: openai.types.Completion = self._make_completion(self.messages)
            return completion.choices[0].message.content

    def chat(self, message, response_lead=None, replace_last=False):
        """
        Add a user message and append + return the assistant's response.
        
        Args:
            message (str): The user message to ChatGPT.
            response_lead (str, optional): A string to prepend to the assistant's response.
                Forces `assistant` response from model to begin with the provided string.
                NOTE: As of March 24, 2023, this does not work with the gpt-4 models.
            replace_last (bool): Replace the last user message and response.
                Useful when you want to retry the last `.chat` call rather than continuing
                the conversation with the previous call as context.
        Returns:
            str: The assistant's response.
        """
        if replace_last:
            self._messages = self._messages[:-2]

        self.user(self.trim(message))
    
        if response_lead is not None:
            self.assistant(response_lead)

        response = self.call()

        # add the response to _messages
        if response_lead is None:
            self.assistant(response)
        else:
            self._messages[-1]['content'] += response

        # return the last message, i.e. the response (plus response lead if applicable)
        return self.messages[-1]['content']


class AzureChatGPT(ChatGPT):

    @property
    def client(self):
        from openai import AzureOpenAI
        model = self.model
        model = model.replace(".", "")
        return AzureOpenAI(azure_deployment=model, api_version="2023-05-15")
