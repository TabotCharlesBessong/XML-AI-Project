import requests
import uuid

subscription_key = '<your-subscription-key>'
endpoint = 'https://api.cognitive.microsofttranslator.com'

# function to translate text from English to French
def translate_text(text):
    path = '/translate?api-version=3.0&from=en&to=fr'
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    params = {
        'api-version': '3.0',
        'from': 'en',
        'to': 'fr'
    }
    body = [{
        'text': text
    }]
    response = requests.post(endpoint + path, params=params, headers=headers, json=body)
    return response.json()[0]['translations'][0]['text']

# example usage
text = 'Hello, how are you?'
translation = translate_text(text)
print(f'Original: {text}')
print(f'Translation: {translation}')