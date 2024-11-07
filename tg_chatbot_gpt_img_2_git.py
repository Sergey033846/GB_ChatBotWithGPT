import telebot
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch
from io import BytesIO
import requests

BOT_TOKEN = 'BOT_TOKEN'

# Токен Hugging Face для авторизации
HUGGINGFACE_TOKEN = 'HUGGINGFACE_TOKEN'

# Тип генерируемого контента (переключатель "image"/"gpt")
content = 'image'

# Авторизация через Hugging Face
login(HUGGINGFACE_TOKEN)

# Инициализация бота
bot = telebot.TeleBot(BOT_TOKEN)

# Подготовка к использованию модели Hugging Face "RuGPT-2 Medium"
gpt_model_name = "sberbank-ai/rugpt3medium_based_on_gpt2"
gpt_model = AutoModelForCausalLM.from_pretrained(gpt_model_name)
tokenizer = AutoTokenizer.from_pretrained(gpt_model_name)
tokenizer.pad_token = tokenizer.eos_token

# Подготовка к использованию API Hugging Face для генерации изображений
img_api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2"
img_api_headers = {
    "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
    "Content-Type": "application/json"
}


# Вспомогательныя функция для генерации изображения с использованием API Hugging Face
def generate_image(prompt):    
    json_payload = {
        "inputs": prompt,
        "options": {"use_gpu": True}
    }
    
    response = requests.post(img_api_url, headers=img_api_headers, json=json_payload)
    
    if response.status_code == 200:
        # Получаем изображение в бинарном формате
        image_data = response.content
        return image_data
    else:
        # Выводим подробную информацию об ошибке
        print(f"Error generating image: {response.status_code}")
        print("Response content:", response.text)
        return None


# Функция для генерации текста
def generate_rugpt2_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Генерация текста с помощью модели RuGPT-2 Medium
    with torch.no_grad():        
        outputs = gpt_model.generate(
            input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, 
            temperature=0.7, top_k=50, top_p=0.95, do_sample=True)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Обработчик команд /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Я — бот, использующий RuGPT-2 Medium для генерации текста и Stable Diffusion 2 для генерации изображений.")
    bot.send_message(message.chat_id, "Напиши мне что-нибудь, и я отвечу!")


# Обработчик команд /image
@bot.message_handler(commands=['image'])
def send_welcome(message):
    global content
    content = 'image'
    bot.reply_to(message, "Отправь мне сообщение, и я постараюсь сгенерировать изображение. Я использую модель Stable Diffusion 2.")


# Обработчик команд /image
@bot.message_handler(commands=['gpt'])
def send_welcome(message):
    global content
    content = 'gpt'
    bot.reply_to(message, "Отправь мне сообщение, и я постараюсь сгенерировать ответ. Я использую модель RuGPT-2 Medium для генерации текста на русском языке.")


# Обработчик команды /help
@bot.message_handler(commands=['help'])
def send_help(message):
    command_list = "/start - запуск бота\n" \
                   "/help - описание бота\n" \
                   "/gpt - генерация текста\n" \
                   "/image - генерация изображения"
    bot.reply_to(message, f"Доступные команды:\n{command_list}")
    bot.send_message(message.chat.id, "Чтобы начать, просто отправьте сообщение.")


# Обработчик текстовых сообщений
@bot.message_handler(func=lambda message: True)
def echo_all(message):    
    prompt = message.text
    
    global content
    if content == 'gpt':
        # Генерируем ответ с помощью RuGPT-2 
        bot.send_message(message.chat.id, "Генерирую текст, пожалуйста, подождите...")        

        response = generate_rugpt2_response(prompt)        
        
        bot.reply_to(message, response)
    elif content == 'image':
        # Генерируем изображение
        bot.send_message(message.chat.id, "Создаю изображение, пожалуйста, подождите...")        
        
        image_data = generate_image(prompt)

        if image_data:            
            image_file = BytesIO(image_data)
            image_file.name = 'generated_image.png'
            image_file.seek(0)        
            bot.send_photo(message.chat.id, image_file)
        else:
            bot.send_message(message.chat.id, "Произошла ошибка при генерации изображения.")


if __name__ == "__main__":
    print("Bot is running...")
    bot.polling()
