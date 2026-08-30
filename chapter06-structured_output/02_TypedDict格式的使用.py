"""
@Author:dym
@Desc: with_structured_output 的 TypedDict 模式 —— 让模型返回 dict 而不是 Pydantic 实例.

langchain 的 model.with_structured_output(MovieTypedDict) 会基于 MovieTypedDict 的
字段名 + type + Annotated[..., description] 自动生成 JSON Schema, 让模型按 schema 输出;
解析后用 pydantic 校验 (内部), 但最终返回的是 dict, 不是 Pydantic 实例.
TypedDict 和 Pydantic BaseModel 的核心区别就是: TypedDict 是纯类型注解, 不做运行时
校验, 输出类型更轻; Pydantic 是带校验的类, 输出是强类型对象.

注意 (踩坑总结, minimax 兼容 OpenAI 协议路上有 4 个暗坑, 本文件实际命中 4 条):
  1. 模型名必须用 minimax api.minimaxi.com/v1 上真实存在的型号.
     当前可用的只有 M 系列: MiniMax-M3 / M2.7 / M2.7-highspeed /
     M2.5 / M2.5-highspeed / M2.1 / M2.1-highspeed / M2.
     写 MiniMax-Text-01 / abab6.5s-chat 这类老名字 = 模型不存在, 服务端会
     静默回退到默认模型, 行为不可预期.
  2. 【minimax structured output 协议限制】字段类型只能用
     ["string","number","boolean","object","array"], 没有 "integer"!
     所以 year 不能用 int, 必须用 float (Pydantic schema 里会生成 type=number).
     OpenAI 区分 integer/number, MiniMax 不区分 —— 这是踩过 2013 错的坑.
  3. M 系列模型默认开启 thinking (推理模式), 会把 <think>...</think> 拼在
     content 里. response_format 解析器看到的就是乱码, 然后报 Invalid JSON.
     解法: extra_body={"thinking": {"type": "disabled"}} 显式关掉.
  4. with_structured_output(schema) 默认走 response_format 路径, minimax M 系列
     对这条路径支持不稳定, 模型经常忽略 schema 输出自然语言. 切到
     method="function_calling" 让 schema 包成伪 tool, 强制 tool_calls 输出,
     是 minimax 上的稳定路径.
  5. function_calling 路径下, minimax M 系列对 OpenAI 的 tool_choice="any"
     字段支持不完整, 仍有概率不调用 tool (finish_reason=stop, tool_calls=[],
     result=None). 实测扁平 TypedDict 约 30% 概率, 嵌套 (List[Actor]) 约 60%.
     客户端必须带 retry, 拿到 None 时自动重试.
"""
import os
from typing import Union
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from typing_extensions import Annotated, TypedDict


# ============================================================
# 1、读取 .env 配置(.env 里的变量优先级最高, override=True 会覆盖同名 shell 环境变量)
# ============================================================
load_dotenv(override=True)

# ============================================================
# 2、参数校验:防止 .env 缺字段时给一个晦涩的 pydantic ValidationError
# ============================================================
API_KEY  = os.getenv("MINIMAX_API_KEY")
BASE_URL = os.getenv("MINIMAX_BASE_URL")
if not API_KEY:
    raise RuntimeError(
        "未找到环境变量 MINIMAX_API_KEY。请在项目根目录 .env 中配置:\n"
        "    MINIMAX_API_KEY=你的密钥\n"
        "    MINIMAX_BASE_URL=https://api.minimaxi.com/v1"
    )
if not BASE_URL:
    raise RuntimeError(
        "未找到环境变量 MINIMAX_BASE_URL。请在 .env 中配置,例如:\n"
        "    https://api.minimaxi.com/v1"
    )

# ============================================================
# 3、初始化模型 —— 直接用 ChatOpenAI 而非 init_chat_model,
#    是因为要透传 extra_body 关掉 minimax M 系列的 thinking 模式,
#    init_chat_model 的 model_kwargs 不会把所有字段透给底层 ChatOpenAI,
#    走 ChatOpenAI 直接传 extra_body 最稳.
# ============================================================
model = ChatOpenAI(
    model="MiniMax-M3",              # api.minimaxi.com/v1 上当前最强 chat 模型
    api_key=API_KEY,
    base_url=BASE_URL,
    extra_body={"thinking": {"type": "disabled"}},  # 关掉 thinking, 避免 <think> 污染 content
)


# ============================================================
# 4、用 TypedDict 定义期望返回的结构.
#    Annotated[type, description] 里的 description 会写进生成的 JSON Schema,
#    模型看到的就是字段名 + type + description.
#    注意: year 必须用 float 而不是 int (踩坑总结 #2, minimax schema 不支持 integer).
# ============================================================
class MovieTypedDict(TypedDict):
    """电影的信息"""
    title    : Annotated[str,   "电影的名称"]
    year     : Annotated[float, "电影的上映年份, 四位数"]   # minimax schema 无 integer, 必须 float
    director : Annotated[str,   "电影的导演"]
    rating   : Annotated[float, "电影的评分, 满分 10 分, 可包含一位小数"]


# ============================================================
# 5、调用层 —— minimax M 系列在 function_calling 路径下有概率返回 None
#    (finish_reason=stop, tool_calls=[]). 包一层轻量 retry, 拿到 None 就重发.
#    与 01_格式化输出_pydantic.py 中的逻辑一致, 本地复制一份避免跨文件 import 耦合.
# ============================================================
def invoke_with_retry(structured_model, prompt: Union[str, list], max_retries: int = 2):
    """with_structured_output 的 result 为 None 时重试, 避免偶发 None 阻塞下游.
    prompt 支持 str 或 [SystemMessage, HumanMessage, ...] 列表."""
    for attempt in range(max_retries + 1):
        result = structured_model.invoke(prompt)
        if result is not None:
            return result
        if attempt < max_retries:
            print(f"[retry] attempt {attempt + 1}/{max_retries} 返回 None, 重试中...")
    raise RuntimeError(
        f"调用 {max_retries + 1} 次均返回 None, minimax M 系列 function_calling "
        f"路径偶发失效, 请检查网络/稍后再试或换模型."
    )


# ============================================================
# 6、结构化输出 —— 与 Pydantic 模式一样, 推荐 method="function_calling"
#    (踩坑总结 #4). TypedDict 没有 runtime 校验, 但 langchain 内部仍用
#    pydantic 校验生成的 schema, 然后反序列化为 dict 返回.
# ============================================================
structured_model = model.with_structured_output(MovieTypedDict, method="function_calling")

response = invoke_with_retry(
    structured_model,
    [
        SystemMessage(content="你是一个信息抽取助手。请始终使用工具返回 JSON, 不要输出任何自然语言解释."),
        HumanMessage(content="给我介绍一下电影《星际穿越》"),
    ],
    max_retries=5,
)

print(response)
print(type(response))


# ============================================================
# 7、嵌套 TypedDict —— TypedDict 同样支持嵌套, 用其他 TypedDict 作为字段类型.
#    嵌套结构 (Movie.actors: List[Actor]) 在 minimax M 系列 function_calling
#    路径下的 tool 失败率明显高于扁平 TypedDict (实测约 60% finish_reason=stop
#    返回自然语言). 两条修复叠加 (与 01_格式化输出_pydantic.py 同款):
#      ① invoke 改用 [SystemMessage, HumanMessage] 而非纯字符串:
#         system message 强制 "请使用工具返回 JSON, 不要输出自然语言",
#         实测能把 8/8 100% 调用 tool (原纯字符串调用 8 次约 5 次失败).
#      ② invoke_with_retry 兜底偶发: max_retries=5 把极端情况压到 ~4.7%.
#    跟扁平 TypedDict 对比, 嵌套输出的整体结构仍然是 dict, 只是嵌套的子字段
#    也是 dict (Actor → dict), 不是 Pydantic 实例.
# ============================================================
class Actor(TypedDict):
    """演员"""
    name      : Annotated[str,   "演员姓名"]
    character : Annotated[str,   "饰演的角色名"]


class MovieWithActors(TypedDict):
    """电影信息(含演员列表)"""
    title  : Annotated[str,                 "电影的名称"]
    year   : Annotated[float,               "电影的上映年份, 四位数"]   # minimax schema 无 integer, 必须 float
    actors : Annotated[list[Actor],         "主演演员列表"]


nested_model = model.with_structured_output(MovieWithActors, method="function_calling")

response_nested = invoke_with_retry(
    nested_model,
    [
        SystemMessage(content="你是一个信息抽取助手。请始终使用工具返回 JSON, 不要输出任何自然语言解释."),
        HumanMessage(content="给我介绍一下电影《星际穿越》, 必须返回 actors 字段, 至少填 3 个主要演员, 每位演员给出姓名和饰演角色"),
    ],
    max_retries=5,
)

print(response_nested)
print(type(response_nested))

# TypedDict 是纯类型注解, 没有 runtime 校验, 嵌套字段 actors 是可选的,
# minimax M 系列在嵌套 schema 上有概率只填部分字段 (实测上面 prompt 不强制时
# 8 次约 7 次漏掉 actors). 这里用 .get() 兜底, 不让 demo 因为偶发缺字段而崩.
actors = response_nested.get("actors", [])
print(f"actors: {actors}  (len={len(actors)})")
if actors:
    print(f"  actors[0] type: {type(actors[0]).__name__}  value: {actors[0]}")

class MovieDict(TypedDict):
    """电影的信息"""
    title    : Annotated[str,   "电影的名称"]
    year     : Annotated[float, "电影的上映时间，四位数"]   # minimax schema 无 integer, 必须 float
    director : Annotated[str,   "电影的导演"]
    rating   : Annotated[float, "电影的评分，满分是10分，可以包含一位小数"]


# 复用上面已经初始化好的 minimax model, 不要再起一个 closeai 的 ChatOpenAI.
# 注意与第 6 节 MovieTypedDict 的区别:
#   - MovieTypedDict (第 6 节): 字段名 title/year/director/rating 跟这里完全一样,
#     演示的是"嵌套用法"——第 6 节没有嵌套, 第 7 节加 actors 才是嵌套.
#   - MovieDict (本节): 跟原 closeai 教程对照, 演示"扁平 TypedDict + 用户 prompt
#     中信息不完整时怎么处理"——评级 rating 不在 prompt 里, schema 允许漏字段,
#     原 closeai 默认 response_format 路径下模型会返回 null; minimax function_calling
#     路径下模型通常也会省略, 所以要 .get() 兜底.
# minimax 兼容要点:
#   - method="function_calling" 强制 tool_calls, 避开默认 response_format 不稳定
#   - year 必须 float (minimax schema 无 integer, 写 int 会变成 number 类型, 行为不可预期)
#   - 不强制要求所有字段必填, 漏字段用 .get() 兜底
structured_model = model.with_structured_output(MovieDict, method="function_calling")
response = invoke_with_retry(
    structured_model,
    "根据这段话抽取盗梦空间的信息，不包含的信息可以留空：盗梦空间在2010年上映，导演是克里斯托弗·诺兰。",
    max_retries=2,
)
print(response)
# minimax M 系列对"信息不全 + 字段可选"的 prompt, 经常省略未提及的字段.
# 原 closeai 用的是 Pydantic 实例访问 (response.year), 必漏则 AttributeError;
# 这里改成 dict + .get(), 漏字段返回 None 而不是 KeyError, 演示 TypedDict 的
# "纯类型注解, 无 runtime 校验" 优势正好体现在这里.
print(f"title   : {response.get('title')}")
print(f"year    : {response.get('year')}")
print(f"director: {response.get('director')}")
print(f"rating  : {response.get('rating')}")
