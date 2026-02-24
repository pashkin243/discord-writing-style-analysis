import discord

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# kanal id - kogumine sees
collecting_channels: set[int] = set()
# kanal id - sõnumite counter
message_counts: dict[int, int] = {}

@client.event
async def on_ready():
    print(f"Logged in as user {client.user}")

@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    
    content = (message.content or "").strip()
    channel_id = message.channel.id

    # --- KÄSUD

    # test - kui vastab, siis bot töötab
    if content.lower() == "!test":
        await message.channel.send("tere tere")
        return
    
    # kogumise käsk. kui sees, saab loa koguda sõnumeid
    if content.lower() == "!collect on":
        collecting_channels.add(channel_id)
        await message.channel.send("Collecting **enabled** in this channel. Type **!collect off** to disable.")
        return

    # kogumise välja lülitamine
    if content.lower() == "!collect off":
        collecting_channels.discard(channel_id)
        await message.channel.send("Collecting **disabled** in this channel. Type **!collect on** to enable.")
        return
    
    # kogumise staatuse kontroll
    if content.lower() == "!collect status":
        status = "ON" if channel_id in collecting_channels else "OFF"
        await message.channel.send(f"Collecting in this channel currently: **{status}**")
        return
    
    # stats - mitu sõnumit kogutud
    if content.lower() == "!stats":
        count = message_counts.get(channel_id, 0)
        await message.channel.send(f"Messages collected in this channel: **{count}**")
        return
    
    # --- KOGUMINE (test, lihtsalt printimine)
    if channel_id in collecting_channels:
        # käskude eemaldamine
        if content.startswith(("!", "/", ".")):
            return
        message_counts[channel_id] = message_counts.get(channel_id, 0) + 1
        print(
            f"[COLLECT #{message_counts[channel_id]}] "
            f"#{message.channel} | {message.author}: {content}",
            flush=True
        )