import openai
import re

openai.api_key = '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'


def set_reward_by_gpt(prompt:str =None):
    prompt = '변동성이 작은 성과를 갖는 투자 전략을 찾아줘' if prompt is None else prompt

    prompt_ = f'투자 성과를 평가하는 reward 함수를 개발하고자 합니다. \
            이 reward 함수는 지도학습 모델을 학습시키기 위한 목적으로 사용될 것입니다. \
            또한 이 reward 함수는 {prompt} 라는 사용자의 요구 사항을 만족해야 합니다. \
            아래 요구사항에 따라 reward 함수 코드를 작성해주세요.\
            (1) 함수 이름: `get_r`.\
            (2) 입력 매개변수: `result`라는 딕셔너리를 입력으로 받습니다.\
            (3) `result`는 다음과 같은 키를 포함합니다:\
                - "sharpe": 샤프 지수 (float)\
                - "expect": 기대 수익률 (float)\
                - "sigma": 수익률의 표준 편차 (float)\
                - "mdd": 최대 낙폭 (float)\
    (4) "sharpe", "expect", "sigma", "mdd" 중 하나 이상을 사용하여 reward를 계산합니다.\
    (5) 최종 reward를 리턴하기 전에, reward를 텐서로 변환합니다: `reward = torch.tensor([reward])`.\
    (6) 함수 코드를 코드 블록으로 감싸서 출력해주세요. (예: ```python ... 코드 ... ```)\
    reward 함수를 작성해 주시고, 함수 사용법과 예제는 제외해 주세요.'

    messages = []
    messages.append({'role':'user', 'content':f"User:{prompt_}"})
    messages.append({'role':'system', 'content':'You are a financial machine learning researcher'})

    response = openai.chat.completions.create(
        model = 'gpt-3.5-turbo',
        messages=messages,
        temperature=1.0,
    )

    chat_response = response.choices[0].message.content

    pattern = r'```python(.*?)```'
    match = re.search(pattern, chat_response, re.DOTALL)

    code_block = match.group(1)

    # 문자열을 파이썬 파일로 저장
    with open('reward.py', 'w') as file:
        file.write(code_block)

    return code_block


prompt = "다음 벡터의 각 원소는 팩터 투자에서 각 팩터의 중요도를 나타낸다. \
        첫번째 인덱스는 3일 주가 모멘텀의 중요도, \
        두번째 인덱스는 7일 주가 모멘텀의 중요도, \
        세번째 인덱스는 14일 주가 모멘텀의 중요도를 나타낸다. \
        입력 받은 예를 들어서 "