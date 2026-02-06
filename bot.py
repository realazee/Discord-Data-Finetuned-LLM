"""
Discord Bot for Fine-tuned TinyLlama
Hosts your fine-tuned model as a Discord user app with slash commands.
"""

import os
import asyncio
import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN not found in .env file!")

# Model Configuration
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./fine_tuned_discord_model" # Path where train.py saved adapters
MAX_NEW_TOKENS = 150  # length of response
TEMPERATURE = 0.8    
TOP_P = 0.9

# ============================================================================
# MODEL INFERENCE
# ============================================================================

class ModelInference:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.loading = False
        self.special_tokens = ["<|user|>", "<|assistant|>", "<|mod|>", "</s>"]

    def load_model(self):
        if self.loaded or self.loading: return
        self.loading = True
        print("📦 Loading model...")

        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            print(f"   Base: {BASE_MODEL_NAME}")
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )

            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"   Adapters: {ADAPTER_PATH}")
            self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
            self.model.eval()

            self.loaded = True
            self.loading = False
            print("✅ Model loaded!")

        except Exception as e:
            self.loading = False
            print(f"❌ Error: {e}")
            raise

    def generate(self, prompt: str) -> str:
        if not self.loaded: return "⚠️ Model loading..."

        formatted_prompt = f"<|user|>\n<|assistant|>\n{prompt}</s>"

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                top_p=TOP_P,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the model's response after <|assistant|>
        if "<|assistant|>" in generated:
            response = generated.split("<|assistant|>")[-1].strip()
        else:
            response = generated

        # Remove any leftover special tokens
        for token in self.special_tokens:
            response = response.replace(token, "")

        # Remove the prompt if it accidentally appears at start
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        return response


model_inference = ModelInference()

# ============================================================================
# DISCORD BOT
# ============================================================================

class MimicBot(commands.Bot):
    def __init__(self):
        # Intents are less important for purely slash command User Apps, but good to have
        intents = discord.Intents.default()
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self):
        print("🔄 Syncing commands...")
        
        # Guild-specific sync is INSTANT (global can take up to 1 hour)
        test_guild_id = os.getenv("TEST_GUILD_ID")
        if test_guild_id:
            guild = discord.Object(id=int(test_guild_id))
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            print(f"✅ Synced to guild {test_guild_id} (instant)")
        
        # Also sync globally (for User App installs, takes longer)
        await self.tree.sync()
        print("✅ Synced globally (may take up to 1 hour to propagate)")
        
        # Load model in background
        asyncio.get_event_loop().run_in_executor(None, model_inference.load_model)

    async def on_ready(self):
        print(f"\n✨ Bot is ready! Logged in as {self.user}")
        print("   Try typing /prompt in Discord!")

bot = MimicBot()

@bot.tree.command(name="prompt", description="Generate a response in your style")
@app_commands.describe(text="Text to respond to")
async def prompt(interaction: discord.Interaction, text: str):
    if not model_inference.loaded:
        await interaction.response.send_message("⏳ Model is loading...", ephemeral=True)
        return

    # Defer (ephemeral=False by default, so it's public)
    await interaction.response.defer()
    
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None, model_inference.generate, text
        )
        if len(response) > 3500: response = response[:3500] + "..."
        
        # Create Embed
        embed = discord.Embed(
            title="fakeazee",  # <-- your desired title
            color=discord.Color.blue()
        )

        embed.add_field(name="Prompt", value=text, inline=False)
        embed.add_field(name="Response", value=response, inline=False)
        embed.set_footer(text=f"Generated by ur mom")

        
        await interaction.followup.send(embed=embed)
    except Exception as e:
        await interaction.followup.send(f"Error: {e}")

@bot.tree.command(name="status", description="Check model status")
async def status(interaction: discord.Interaction):
    status = "🟢 Ready" if model_inference.loaded else "🔴 Loading..."
    await interaction.response.send_message(f"Status: {status}")

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
