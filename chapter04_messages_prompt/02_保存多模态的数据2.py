import base64

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv(override=True)

model = init_chat_model(
    model="MiniMax-M3",
    model_provider="openai",
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL")
)

def encode_image(img_path):
    """将一张本地图片转换成 Base64 编码的 Data URI 字符串,方便在文本中嵌入图片数据"""
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# 图像路径
img_path = "image_test.png"

# 获取图像base64编码字符串
base64_image = encode_image(img_path)

response = model.invoke(
    [
        # 此种格式可用
        # HumanMessage(
        #     content=[
        #         {'type': 'text', 'text': '这张图里有什么？'},
        #         {
        #             'type': 'image_url',
        #             "image_url": base64_image,
        #         }
        #     ]
        # )
        # 推荐的统一写法
        HumanMessage(
            content_blocks=[
                {'type': 'text', 'text': '这张图里有什么？'},
                {
                    'type': 'image',
                    'base64': base64_image,
                    'mime_type': 'image/png',
                }
            ]
        )

    ]
)
print(response.content)