"""Skill — 프로젝트의 모든 Skill을 여기서 관리한다.

Skill = 툴 묶음 + 사용 지침(prompt). 툴만 바인딩하면 "언제 쓰는 툴인지"를
에이전트마다 SystemMessage에 다시 적게 된다. 지침을 Skill에 두면 한 곳에서 고친다.

    llm = init_chat_model("main").bind_skills([clock_kst])
"""

from klafi.tool import Skill

from .tools import kst_now

clock_kst = Skill(
    name="clock_kst",
    tools=[kst_now],
    prompt="한국 시각이 필요하면 kst_now 툴을 사용한다.",
)
