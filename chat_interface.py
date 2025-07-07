#!/usr/bin/env python3
"""
Interactive Chat Interface for ChatDev
=====================================

This module provides an interactive chat interface that allows users to discuss
their project requirements with an AI agent before starting the development process.

The chat agent helps users:
- Clarify project requirements
- Define scope and features
- Specify technical preferences
- Generate detailed task prompts for the main ChatDev system
"""

import os
import sys
from typing import Dict, Optional, List
from camel.typing import ModelType
from camel.messages import SystemMessage
from camel.agents.chat_agent import ChatAgent
from camel.messages import ChatMessage, MessageType
from camel.typing import RoleType
from llm_interface import call_llm


class ProjectChatAgent:
    """
    Specialized chat agent for discussing project requirements.
    """
    
    def __init__(self, model_type: ModelType = ModelType.GEMINI_2_0_FLASH_EXP):
        self.model_type = model_type
        self.conversation_history = []
        self.project_details = {
            'task': '',
            'name': '',
            'features': [],
            'tech_preferences': [],
            'clarifications': []
        }
        
    def get_system_prompt(self) -> str:
        """Get the system prompt for the project discussion agent."""
        return """You are a helpful AI assistant specialized in software project planning and requirement gathering. Your role is to help users clearly define their software project requirements through natural conversation, then guide them to start development.

Your goals:
1. Understand what the user wants to build
2. Ask clarifying questions to get specific details
3. Help define the scope and key features
4. Gather technical preferences if relevant
5. Guide users toward starting development when requirements are clear

Guidelines:
- Be conversational and friendly
- Ask one question at a time to avoid overwhelming the user
- Focus on practical, implementable features
- Help users think through edge cases and requirements
- When the user indicates they want to start (using words like "start", "begin", "go", "start working", etc.), acknowledge their readiness and confirm the requirements
- If requirements seem sufficient, encourage them to proceed with development
- You are NOT building the software yourself - you're gathering requirements for a development system

Important: When users say they want to "start", "begin development", or similar phrases, recognize this as their signal to move forward with the project. Don't discourage them or say you can't build it - you're preparing requirements for the development system.

Keep responses concise but helpful. Don't write code or technical implementations - focus on understanding requirements and guiding toward development start."""

    def generate_response(self, user_input: str) -> str:
        """Generate a response using the LLM interface."""
        # Build conversation context
        messages = [
            {"role": "system", "content": self.get_system_prompt()}
        ]
        
        # Add conversation history
        for msg in self.conversation_history[-10:]:  # Keep last 10 messages for context
            messages.append(msg)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = call_llm(messages, model="gemini-2.0-flash-exp", temperature=0.7)
            
            if response and "choices" in response:
                assistant_response = response["choices"][0]["message"]["content"]
                
                # Update conversation history
                self.conversation_history.append({"role": "user", "content": user_input})
                self.conversation_history.append({"role": "assistant", "content": assistant_response})
                
                return assistant_response
            else:
                return "I'm sorry, I'm having trouble processing your request. Could you please try again?"
                
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm experiencing some technical difficulties. Let's continue our conversation."

    def extract_project_details(self) -> Dict[str, str]:
        """Extract final project details from the conversation."""
        # Use LLM to analyze the conversation and extract key details
        analysis_prompt = f"""Based on the following conversation about a software project, extract the key project details:

Conversation:
{self._format_conversation_for_analysis()}

Please provide:
1. A clear, detailed task description (2-3 sentences) that describes what software to build
2. A suitable project name (1-3 words, no spaces, use underscores if needed)

Format your response as:
TASK: [detailed task description]
NAME: [project_name]

Make sure the task description is specific enough for a development team to understand what to build."""

        try:
            response = call_llm(analysis_prompt, model="gemini-2.0-flash-exp", temperature=0.3)
            
            if response and "choices" in response:
                content = response["choices"][0]["message"]["content"]
                
                # Parse the response
                task = ""
                name = ""
                
                for line in content.split('\n'):
                    if line.startswith('TASK:'):
                        task = line.replace('TASK:', '').strip()
                    elif line.startswith('NAME:'):
                        name = line.replace('NAME:', '').strip()
                
                return {
                    'task': task if task else "Develop a software application based on user requirements.",
                    'name': name if name else "UserProject"
                }
            
        except Exception as e:
            print(f"Error extracting project details: {e}")
        
        # Fallback
        return {
            'task': "Develop a software application based on user requirements.",
            'name': "UserProject"
        }

    def _format_conversation_for_analysis(self) -> str:
        """Format conversation history for analysis."""
        formatted = []
        for msg in self.conversation_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)


class ChatInterface:
    """
    Main chat interface for interactive project requirement gathering.
    """
    
    def __init__(self, model_type: ModelType = ModelType.GEMINI_2_0_FLASH_EXP):
        self.model_type = model_type
        self.chat_agent = ProjectChatAgent(model_type)
        
    def start_conversation(self) -> Optional[Dict[str, str]]:
        """
        Start the interactive conversation.
        
        Returns:
            Dict with 'task' and 'name' keys if successful, None if user exits
        """
        print("\n🤖 Hi! I'm here to help you plan your software project.")
        print("💬 Tell me what you'd like to build, and I'll help you refine the requirements.")
        print("🚀 When you're ready, just say 'start' and I'll begin the development process!")
        print("💡 You can also type 'exit' to quit anytime.\n")
        
        # Initial greeting
        initial_response = self.chat_agent.generate_response(
            "The user wants to start a new software project. Greet them and ask what they'd like to build."
        )
        print(f"🤖 {initial_response}\n")
        
        while True:
            try:
                user_input = input("👤 You: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\n👋 Thanks for chatting! Feel free to come back anytime.")
                    return None
                    
                # Check for various ways user might indicate they want to start
                start_keywords = ['start', 'begin', 'go', 'start development', 'start it', 'start working',
                                'lets start', "let's start", 'ready to start', 'start now', 'proceed']

                if any(keyword in user_input.lower() for keyword in start_keywords):
                    print("\n🔄 Analyzing our conversation to generate project details...")
                    project_details = self.chat_agent.extract_project_details()

                    # Show summary and confirm
                    print(f"\n📋 Project Summary:")
                    print(f"   Name: {project_details['name']}")
                    print(f"   Task: {project_details['task']}")

                    confirm = input("\n✅ Does this look correct? (y/n): ").strip().lower()
                    if confirm in ['y', 'yes', 'ok', 'correct', '']:  # Empty input defaults to yes
                        return project_details
                    else:
                        print("Let's continue refining the requirements...")
                        continue
                
                # Generate response
                response = self.chat_agent.generate_response(user_input)
                print(f"\n🤖 {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                return None
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                print("Let's try to continue...")
                continue
