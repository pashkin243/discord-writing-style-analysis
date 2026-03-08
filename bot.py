import discord
import db

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# kanal id - kogumine sees
collecting_channels: set[int] = set()
# kanal id - sõnumite counter
message_counts: dict[int, int] = {}

# maksimaalne lugemine (tagasi)
MAX_BACKFILL = 1000
# abifunktsioon
def should_collect_text(content: str) -> bool:
    c = (content or "").strip()
    if not c:
        return False
    if c.startswith(("!", "/", ".")):
        return False
    return True

@client.event
async def on_ready():
    await db.init_db()
    print(f"Logged in as user {client.user}", flush=True)
    print("DB ready", flush=True)

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
    
    #help käsk - prindib välja, mis käsud on.
    if content.lower() == "!help":
        await message.channel.send(
            "Here is a list of my commands:\n\n" \
            "**!collect on**\n" \
            "Turns on active collecting of **new** messages sent in the channel.\n\n" \
            "**!collect off**\n" \
            "Turns off the active collecting of messages in the channel.\n\n" \
            "**!collect status**\n" \
            "Displays whether active collecting of messages in the channel is ON or OFF.\n\n" \
            "**!collect last [number]**\n" \
            "Collects the last X amount of messages sent in this channel. Maximum is 1000.\n\n" \
            "**!stats**\n" \
            "Shows the total number of messages collected from this channel.\n\n" \
            "**!profile**\n" \
            "Shows several statistics for the messages sent in this channel. \n\n" \
            "**!profile @User**\n" \
            "Shows several statistics for the messages sent in this channel by the mentioned user."
        )
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
        count = await db.count_messages(channel_id)
        await message.channel.send(f"Messages collected in this channel: **{count}**")
        return
    
    # viimaste sõnumite kogumine
    if content.lower().startswith("!collect last"):
        parts = content.split()
        # vea handling
        if len(parts) != 3:
            await message.channel.send("Invalid command. Try: **!collect last [number]** (example: *!collect last 150*)")
            return
        
        try:
            n = int(parts[2])
        except ValueError:
            await message.channel.send("Invalid command. Provide a valid number.")
            return
        
        if n < 1:
            await message.channel.send("Number must be at least 1.")
            return
        
        if n > MAX_BACKFILL:
            await message.channel.send(f"Error: Maximum number of messages is {MAX_BACKFILL}.")
            return
        
        collected = 0
        # fetchib kanali ajaloo, seatud limiidiga
        async for msg in message.channel.history(limit=n):
            if msg.author.bot:
                continue # ei kogu boti saadetut
            if msg.id == message.id:
                continue # ei kogu käsusõnumit
            text = (msg.content or "").strip()
            if not should_collect_text(text):
                continue
            inserted = await db.insert_message(
                message_id=msg.id,
                guild_id=msg.guild.id if msg.guild else None,
                channel_id=channel_id,
                author_id=msg.author.id,
                content=text,
                created_at=msg.created_at,
            )
            if not inserted:
                continue

            message_counts[channel_id] = message_counts.get(channel_id, 0) + 1
            collected += 1

            print(
                f"[BACKFILL #{message_counts[channel_id]}] "
                f"#{msg.channel} | {msg.author}: {text}",
                flush=True
            )
        
        await message.channel.send(f"Backfilled **{collected}** messages from the last **{n}** in this channel.")
        return
    
    #!profile, näitab kanali või isiku sõnumite statistikat
    if content.lower().startswith("!profile"):
        target_user = None
        # kas kasutaja on mainitud?
        if message.mentions:
            target_user = message.mentions[0]
        if target_user:
            profile = await db.get_profile(channel_id, target_user.id)
            name = target_user.display_name
        else:
            profile = await db.get_profile(channel_id)
            name = f"#{message.channel.name}"
        if profile is None:
            await message.channel.send("No messages found!")
            return
        await message.channel.send(
            f"Profile for **{name}**\n\n"
            f"Total messages: **{profile['messages']}**\n"
            f"Average message length: **{profile['avg_length']:.1f} characters**\n"
            f"Average words per message: **{profile['avg_words']:.1f}**\n"
            f"Exclamation marks per message: **{profile['exclamations_per_msg']:.2f}**\n"
            f"Question marks per message: **{profile['questions_per_msg']:.2f}**\n"
            f"Uppercase ratio: **{profile['uppercase_ratio']:.2f}**"
        )
        return

    # --- KOGUMINE (test, lihtsalt printimine)
    if channel_id in collecting_channels:
        # käskude eemaldamine
        if not should_collect_text(content):
            return
        # db jaoks plokk
        inserted = await db.insert_message(
            message_id=message.id,
            guild_id=message.guild.id if message.guild else None,
            channel_id=channel_id,
            author_id=message.author.id,
            content=content,
            created_at=message.created_at,
        )
        if not inserted:
            return
        # lugemine
        message_counts[channel_id] = message_counts.get(channel_id, 0) + 1
        print(
            f"[COLLECT #{message_counts[channel_id]}] "
            f"#{message.channel} | {message.author}: {content}",
            flush=True
        )