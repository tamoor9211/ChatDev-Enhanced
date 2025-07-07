# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import argparse
import logging
import os
import sys

from camel.typing import ModelType

root = os.path.dirname(__file__)
sys.path.append(root)

from chatdev.chat_chain import ChatChain

# Google AI Studio API is now used instead of OpenAI
google_ai_available = True
try:
    import google.generativeai as genai
except ImportError:
    google_ai_available = False
    print(
        "Warning: Google Generative AI package not found. \n "
        "Please install it with: pip install google-generativeai \n "
        "Make sure to set your GOOGLE_API_KEY environment variable.")


def get_config(company):
    """
    return configuration json files for ChatChain
    user can customize only parts of configuration json files, other files will be left for default
    Args:
        company: customized configuration name under CompanyConfig/

    Returns:
        path to three configuration jsons: [config_path, config_phase_path, config_role_path]
    """
    config_dir = os.path.join(root, "CompanyConfig", company)
    default_config_dir = os.path.join(root, "CompanyConfig", "Default")

    config_files = [
        "ChatChainConfig.json",
        "PhaseConfig.json",
        "RoleConfig.json"
    ]

    config_paths = []

    for config_file in config_files:
        company_config_path = os.path.join(config_dir, config_file)
        default_config_path = os.path.join(default_config_dir, config_file)

        if os.path.exists(company_config_path):
            config_paths.append(company_config_path)
        else:
            config_paths.append(default_config_path)

    return tuple(config_paths)


parser = argparse.ArgumentParser(description='argparse')
parser.add_argument('--config', type=str, default="Default",
                    help="Name of config, which is used to load configuration under CompanyConfig/")
parser.add_argument('--org', type=str, default="DefaultOrganization",
                    help="Name of organization, your software will be generated in WareHouse/name_org_timestamp")
parser.add_argument('--task', type=str, default="",
                    help="Prompt of software (if empty, will start chat interface)")
parser.add_argument('--name', type=str, default="Gomoku",
                    help="Name of software, your software will be generated in WareHouse/name_org_timestamp")
parser.add_argument('--model', type=str, default="GEMINI_2_0_FLASH_EXP",
                    help="Model, choose from {'GEMINI_1_5_FLASH', 'GEMINI_1_5_PRO', 'GEMINI_1_0_PRO', 'GEMINI_PRO', 'GEMINI_PRO_VISION', 'GEMINI_2_0_FLASH_EXP', 'GPT_3_5_TURBO', 'GPT_4', 'GPT_4_TURBO', 'GPT_4O', 'GPT_4O_MINI'}")
parser.add_argument('--path', type=str, default="",
                    help="Your file directory, ChatDev will build upon your software in the Incremental mode")
parser.add_argument('--chat', action='store_true',
                    help="Start interactive chat interface to discuss project requirements")
args = parser.parse_args()

# Define model type mapping
args2type = {'GEMINI_1_5_FLASH': ModelType.GEMINI_1_5_FLASH,
             'GEMINI_1_5_PRO': ModelType.GEMINI_1_5_PRO,
             'GEMINI_1_0_PRO': ModelType.GEMINI_1_0_PRO,
             'GEMINI_PRO': ModelType.GEMINI_PRO,
             'GEMINI_PRO_VISION': ModelType.GEMINI_PRO_VISION,
             'GEMINI_2_0_FLASH_EXP': ModelType.GEMINI_2_0_FLASH_EXP,
             # Keep old names for backward compatibility
             'GPT_3_5_TURBO': ModelType.GEMINI_1_5_FLASH,
             'GPT_4': ModelType.GEMINI_1_5_PRO,
             'GPT_4_TURBO': ModelType.GEMINI_1_5_PRO,
             'GPT_4O': ModelType.GEMINI_1_5_PRO,
             'GPT_4O_MINI': ModelType.GEMINI_1_5_FLASH,
             }

# Check if we should start chat interface
if args.chat or not args.task.strip():
    from chat_interface import ChatInterface

    print("🤖 Welcome to ChatDev Interactive Mode!")
    print("Let's discuss your project requirements...")
    print("=" * 50)

    chat_interface = ChatInterface(model_type=args2type[args.model])
    project_details = chat_interface.start_conversation()

    if project_details:
        # Update args with details from chat
        args.task = project_details['task']
        args.name = project_details['name']
        print(f"\n✅ Great! Starting development of: {args.name}")
        print(f"📝 Task: {args.task}")
        print("=" * 50)
    else:
        print("👋 Chat session ended. Goodbye!")
        sys.exit(0)

# Start ChatDev

# ----------------------------------------
#          Init ChatChain
# ----------------------------------------
config_path, config_phase_path, config_role_path = get_config(args.config)

chat_chain = ChatChain(config_path=config_path,
                       config_phase_path=config_phase_path,
                       config_role_path=config_role_path,
                       task_prompt=args.task,
                       project_name=args.name,
                       org_name=args.org,
                       model_type=args2type[args.model],
                       code_path=args.path)

# ----------------------------------------
#          Init Log
# ----------------------------------------
logging.basicConfig(filename=chat_chain.log_filepath, level=logging.INFO,
                    format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%Y-%d-%m %H:%M:%S', encoding="utf-8")

# ----------------------------------------
#          Pre Processing
# ----------------------------------------

chat_chain.pre_processing()

# ----------------------------------------
#          Personnel Recruitment
# ----------------------------------------

chat_chain.make_recruitment()

# ----------------------------------------
#          Chat Chain
# ----------------------------------------

chat_chain.execute_chain()

# ----------------------------------------
#          Post Processing
# ----------------------------------------

chat_chain.post_processing()