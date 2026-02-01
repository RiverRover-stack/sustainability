"""Agent package - AI-powered energy advisor."""

import sys, os
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_pkg_dir)
if _pkg_dir not in sys.path: sys.path.insert(0, _pkg_dir)
if _src_dir not in sys.path: sys.path.insert(0, _src_dir)

from energy_agent import EnergyAdvisorAgent, get_energy_agent, chat_with_agent
from knowledge_base import get_knowledge_base, query_knowledge_base
from quick_answers import get_quick_answer, get_all_topics
from user_analysis import compute_user_summary, format_user_context, generate_insights

__all__ = [
    'EnergyAdvisorAgent', 'get_energy_agent', 'chat_with_agent',
    'get_knowledge_base', 'query_knowledge_base',
    'get_quick_answer', 'get_all_topics',
    'compute_user_summary', 'format_user_context', 'generate_insights'
]
