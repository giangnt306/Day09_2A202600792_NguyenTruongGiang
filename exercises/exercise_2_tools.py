"""Bài Tập 2: Thêm Tools và Knowledge Base

Hoàn thành các TODO để thêm tool và knowledge base entry mới.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from common.llm import get_llm

# Knowledge base
LEGAL_KNOWLEDGE = [
    {
        "id": "ucc_breach",
        "keywords": ["breach", "contract", "remedies", "damages", "ucc"],
        "text": (
            "Under the Uniform Commercial Code (UCC) Article 2, remedies for breach of contract "
            "include: (1) expectation damages; (2) consequential damages; (3) specific performance; "
            "(4) cover damages. Statute of limitations is typically 4 years (UCC § 2-725)."
        ),
    },
    {
        "id": "labor_law",
        "keywords": ["lao động", "sa thải", "thôi việc", "hợp đồng lao động", "bồi thường", "trợ cấp"],
        "text": (
            "Theo Bộ luật Lao động Việt Nam 2019: (1) Sa thải trái luật phải bồi thường ít nhất 2 tháng lương "
            "và hoàn trả nguyên trạng; (2) Thời hiệu khởi kiện tranh chấp lao động cá nhân là 1 năm kể từ "
            "ngày phát hiện hành vi vi phạm; (3) Trợ cấp thôi việc: 1/2 tháng lương cho mỗi năm làm việc "
            "(áp dụng với hợp đồng không xác định thời hạn bị chấm dứt đúng luật)."
        ),
    },
]


@tool
def search_legal_knowledge(query: str) -> str:
    """Tìm kiếm trong knowledge base pháp lý."""
    query_lower = query.lower()
    for entry in LEGAL_KNOWLEDGE:
        if any(kw in query_lower for kw in entry["keywords"]):
            return f"[{entry['id']}] {entry['text']}"
    return "Không tìm thấy thông tin liên quan."


@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án."""
    statutes = {
        "hợp đồng": "4 năm (UCC § 2-725 với hợp đồng thương mại); 3 năm theo BLDS Việt Nam 2015 (Điều 429).",
        "contract": "4 năm (UCC § 2-725 với hợp đồng thương mại); 3 năm theo BLDS Việt Nam 2015 (Điều 429).",
        "lao động": "1 năm kể từ ngày phát hiện hành vi vi phạm (Bộ luật Lao động 2019, Điều 190).",
        "labor": "1 năm kể từ ngày phát hiện hành vi vi phạm (Bộ luật Lao động 2019, Điều 190).",
        "dân sự": "3 năm kể từ ngày người có quyền biết hoặc phải biết quyền bị xâm phạm (BLDS 2015, Điều 429).",
        "civil": "3 năm kể từ ngày người có quyền biết hoặc phải biết quyền bị xâm phạm (BLDS 2015, Điều 429).",
        "hình sự": "Từ 5–20 năm hoặc không giới hạn tùy mức hình phạt (BLHS 2015, Điều 27).",
        "criminal": "Từ 5–20 năm hoặc không giới hạn tùy mức hình phạt (BLHS 2015, Điều 27).",
        "hành chính": "1 năm kể từ ngày nhận được hoặc biết được quyết định hành chính (Luật TTHC 2015, Điều 116).",
    }
    case_lower = case_type.lower()
    for key, value in statutes.items():
        if key in case_lower:
            return f"Thời hiệu khởi kiện ({case_type}): {value}"
    return f"Không có dữ liệu thời hiệu cho loại vụ án '{case_type}'. Vui lòng tham khảo luật sư."


async def main():
    load_dotenv()
    llm = get_llm()
    
    # TODO: Thêm tool mới vào danh sách
    tools = [search_legal_knowledge, check_statute_of_limitations]
    llm_with_tools = llm.bind_tools(tools)
    
    question = "Thời hiệu khởi kiện vụ vi phạm hợp đồng là bao lâu?"
    
    messages = [
        SystemMessage(content="Bạn là chuyên gia pháp lý. Sử dụng tools để tra cứu thông tin."),
        HumanMessage(content=question),
    ]
    
    print(f"Câu hỏi: {question}\n")
    
    # First LLM call - decide which tools to use
    response = await llm_with_tools.ainvoke(messages)
    messages.append(response)
    
    # Execute tools if requested
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"🔧 Gọi tool: {tool_call['name']}")
            tool_result = None
            
            if tool_call["name"] == "search_legal_knowledge":
                tool_result = search_legal_knowledge.invoke(tool_call["args"])
            elif tool_call["name"] == "check_statute_of_limitations":
                tool_result = check_statute_of_limitations.invoke(tool_call["args"])
            
            if tool_result:
                messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
        
        # Second LLM call - synthesize final answer
        final_response = await llm_with_tools.ainvoke(messages)
        print(f"\n✅ Kết quả:\n{final_response.content}")
    else:
        print(f"\n✅ Kết quả:\n{response.content}")


if __name__ == "__main__":
    asyncio.run(main())
