import os
import discord

TOKEN = os.getenv("DISCORD_TOKEN")

if not TOKEN:
    raise RuntimeError("DISCORD_TOKEN is not set")

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"Logitud sisse kasutajana {client.user}")

@client.event
async def on_message(message):
    if message.author.bot:
        return

    # testimiskäsk - kas töötab?
    if message.content == "!tere":
        await message.channel.send("tere tere")

client.run(TOKEN)