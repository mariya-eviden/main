"""
LLM Augmented Autonomous Agent for Maintenance

Copyright (C) 2023 Eviden. All rights reserved
"""

from datetime import datetime
from functools import cached_property
from pathlib import Path
from textwrap import dedent
from typing import Tuple

import pandas as pd
import streamlit as st
from devtools import debug
from langchain import hub
from langchain.agents import (
    AgentExecutor,
)
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BaseTool, tool
from langchain.tools.retriever import create_retriever_tool
from langchain.vectorstores.base import VectorStore
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.document_loaders import TextLoader
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from loguru import logger
from pydantic import BaseModel, Field

# from pydantic import BaseModel, Field
from python.ai_core.agents_builder import get_agent_builder
from python.ai_core.embeddings import EmbeddingsFactory
from python.ai_core.llm import LlmFactory
from python.ai_core.prompts import dedent_ws, def_prompt
from python.ai_core.vector_store import VectorStoreFactory
from python.config import get_config_str
from python.demos.maintenance_agent.maintenance_data import dummy_database

# Tools setup
PROCEDURES = [
    "procedure_turbine.txt",
    "procedure_generator.txt",
    "procedure_cooling_system.txt",
]

DATA_PATH = Path(get_config_str("documents", "base")) / "maintenance"

VECTOR_STORE_ID = "InMemory"

@st.cache_resource(show_spinner=True)
def maintenance_procedure_vectors(text: str) -> VectorStore:
    """Index and store a document for similarity matching (through embeddings).

    Embeddings are stored in a vector-store.
    """

    vs_factory = VectorStoreFactory(
        id=VECTOR_STORE_ID,
        collection_name="maintenance_procedure",
        embeddings_factory=EmbeddingsFactory(),
    )

    loader = TextLoader(str(DATA_PATH / text))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    vs_factory.add_documents(texts)
    return vs_factory.vector_store


class MaintenanceAgent(BaseModel):
    """Maintenance Agent class"""

    llm_factory: LlmFactory
    embeddings_factory: EmbeddingsFactory
    tools: list[BaseTool] = Field(default_factory=list)
    _agent_executor: AgentExecutor | None = None

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True
        ignored_types = (cached_property,)

    def create_sql_agent(self) -> AgentExecutor:
        """Create an agent for Text-2-SQL.

        Inspired by: https://python.langchain.com/docs/use_cases/qa_structured/sql
        """
        # fmt: off
        few_shots = {
            "Tasks assigned to employee 'employee_name' between '2023-10-22' and '2023-10-28'.": 
                """SELECT task FROM "tasks" WHERE employee = 'employee_name' AND start_date > '2023-10-22' and end_date  < '2023-10-28';""",
            "List of procedures that employee 'employee_name'' knows. ": 
                """SELECT  DISTINCT employee, procedure FROM tasks WHERE employee = 'employee_name';"""
        }

        # fmt: on

        few_shot_docs = [
            Document(page_content=question, metadata={"sql_query": few_shots[question]})
            for question in few_shots.keys()
        ]

        vector_db = VectorStoreFactory(
            id=VECTOR_STORE_ID,
            collection_name="test_maintenance",
            embeddings_factory=self.embeddings_factory,
        ).vector_store
        vector_db.add_documents(few_shot_docs)

        tool_description = """
            This tool will help you understand similar examples to adapt them to the user question.
            Input to this tool should be the user question.
            """
        retriever_tool = create_retriever_tool(
            vector_db.as_retriever(search_kwargs={"k": 1}),
            name="sql_get_similar_examples",
            description=dedent_ws(tool_description),
        )

        custom_suffix = """
            If the query involve time I should get the current time.
            I should first get the similar examples I know. 
            If an example is enough to construct the query, then I build it by adapting values according to actual requests and I execute it.
            Otherwise, I look at the tables in the database to see what I can query, and I should query the schema of the most relevant tables.
            I execute the SQL query to  to retrieve the information.
            """  # noqa: F841

        db = SQLDatabase.from_uri(dummy_database())

        from langchain_community.agent_toolkits.sql.prompt import SQL_PREFIX

        prefix = (
            SQL_PREFIX
            + f"\nCurrent date is: {datetime.now().isoformat()}. DO NOT use any SQL date functions like NOW(), DATE(), DATETIME()"
        )

        llm = self.llm_factory.get()
        return create_sql_agent(
            llm=llm,
            toolkit=SQLDatabaseToolkit(db=db, llm=llm),
            agent_type="tool-calling",
            prefix=prefix,
            verbose=False,
            extra_tools=[retriever_tool],
            # suffix=dedent_ws(custom_suffix),
            max_iterations=5,
        )

    def create_tools(self):
        """Create the tools for our Agent"""

        logger.info("create tools")
        # query_database = Tool(
        #     name="query_maintenance_planning_database",
        #     func=self.create_sql_agent().invoke,
        #     description=(
        #         "Useful for when you need  to answer questions about tasks assigned to employees."
        #         "Input is a question about task planning. It should contain full context"
        #     ),
        # )

        @tool
        def get_planning_info(query: str):
            """Useful for when you need  to answer questions about tasks assigned to employees.
            Input is a question about task planning. It should contain full context"
            """
            return self.create_sql_agent().run(query)

        @tool
        def get_current_time() -> str:
            """A tool to get the current time"""
            return datetime.now().isoformat()

        @tool
        def get_maintenance_times(area: str) -> str:
            """Useful to searches the list of previous maintenance time for the given area.
            Returns a list of date and hours"""
            result = """
            -29/05/2018  13:45
            -16/05/2019  10:00
            -22/07/2020  9:45
            """
            return dedent_ws(result)

        @tool
        def get_maintenance_issues(area: str, time: str) -> str:
            """Searches for the issues during the maintenance of a given area at a given time"""
            result = f"""
            Big big problem in area {area} at {time}
            """
            return dedent(result)

        @tool
        def maintenance_procedure_retriever(full_query: str) -> str:
            """
            Answer to any questions about maintenance procedures, such as tasks, prerequisite, spare parts, required tools etc.
            Input should contain the question with the full context.
            The questions can ask for information about tasks, spare parts, tools etc.
            """

            system_prompt = (
                "Use the given context to answer the question. If you don't know the answer, say you don't know. "
                "Use three sentence maximum and keep the answer concise. \n"
                "Context: {context}"
            )
            # See https://www.linkedin.com/pulse/beginners-guide-retrieval-chain-using-langchain-vijaykumar-kartha-kuinc/
            prompt = def_prompt(system_prompt, "{input}")
            retriever = maintenance_procedure_vectors(PROCEDURES[0]).as_retriever()
            question_answer_chain = create_stuff_documents_chain(
                self.llm_factory.get(), prompt
            )
            chain = create_retrieval_chain(retriever, question_answer_chain)
            return chain.invoke({"input": full_query})

        # https://vijaykumarkartha.medium.com/beginners-guide-to-creating-ai-agents-with-langchain-eaa5c10973e6

        retriever_tool = create_retriever_tool(
            retriever=maintenance_procedure_vectors(PROCEDURES[0]).as_retriever(),
            name="bla",
            description=dedent_ws(
                """
                Answer to any questions about maintenance procedures, such as tasks, prerequisite, spare parts, required tools etc.
                Input should contain full context.
                The questions can ask for information about several tasks, spare parts, tools etc.
                """
            ),
        )

        @tool
        def get_info_from_erp(tools_name: list[str], time: str) -> str:
            """Useful to answer questions about tools, such as availability, localization, ...
            Input should be a list of tools for which such information are requested"""

            # Mock function . A real function would call an ERP or PLM API
            result = ""
            for too_name in tools_name:
                availability = too_name[0].lower() < "r"
                hash = str(too_name.__hash__())
                localization = f"R{hash[1:3]}L{hash[3:7]}"
                result += f"{too_name}:\n\t- availability: {availability} \n\t- localization:{localization}\n"
            return result

        @tool
        def get_sensor_values(
            sensor_name: list[str], start_time: str, end_time: str
        ) -> str:
            """Useful to know the values of a given sensor during a time range."""
            debug(sensor_name, start_time, end_time)
            ls = ",".join([f"'{n}'" for n in sensor_name])
            db = SQLDatabase.from_uri(dummy_database())
            sql = f"""SELECT date, sensor, value, unit FROM  sensor_data 
                      WHERE  sensor IN ({ls}) and date >= '{start_time}' and date <= '{end_time}'"""
            debug(sql)
            result = db.run(sql)
            return str(result)  # #TODO : Correct and  test

        # https://python.langchain.com/docs/modules/agents/agent_types/structured_chat

        tools = [
            get_maintenance_times,
            get_maintenance_issues,
            get_planning_info,
            get_current_time,
            maintenance_procedure_retriever,
            get_info_from_erp,
            get_sensor_values,
        ]
        self.tools = tools

    def add_tools(self, new_tools: list[BaseTool]):
        tool_descriptions = [tool.description for tool in self.tools]
        self.tools += [
            tool for tool in new_tools if tool.description not in tool_descriptions
        ]

    def run(
        self,
        query: str,
        verbose: bool = False,
        extra_callbacks: list[BaseCallbackHandler] | None = None,
        extra_metadata: dict | None = None,
    ) -> Tuple[str, pd.DataFrame | None]:
        """Run the maintenance agent."""

        # info = get_llm_info(self.llm_factory.get_id())
        info = self.llm_factory.info

        agent_builder = get_agent_builder("tool_calling")
        prompt = hub.pull(agent_builder.hub_prompt)

        agent_creator = agent_builder.create_function

        system = dedent_ws(
            """
            You are a helpful assistant to help a maintenance engineer. 
            To do so, you have different tools (functions)  to access maintenance planning, spares etc.
            Make sure to use only the provided tools (functions) to answer the user request.
            If you don't find relevant tool, answer "I don't know"
            """
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # Construct the Tools agent
        llm = self.llm_factory.get()
        agent = agent_creator(llm, self.tools, prompt)  # type: ignore
        # agent = create_tool_calling_agent(llm, self.tools, prompt)

        extra_callbacks = extra_callbacks or []
        # callbacks = app_conf().callback_handlers + extra_callbacks
        callbacks = extra_callbacks

        metadata = {"agentName": "MaintenanceAgent1"}
        metadata |= extra_metadata or {}

        self._agent_executor = AgentExecutor(
            agent=agent,  # type: ignore
            tools=self.tools,
            verbose=verbose,
            handle_parsing_errors=True,
            # callbacks=callbacks,
            # callbacks=app_conf().callback_handlers,
            metadata=metadata,
        )

        cfg = RunnableConfig()
        cfg["callbacks"] = callbacks
        cfg["metadata"] = metadata
        result = self._agent_executor.invoke({"input": query}, cfg)
        return result["output"]


# python_repl._globals['df'] = df
