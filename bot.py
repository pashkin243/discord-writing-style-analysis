import discord
import db
import style

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

#abifunktsioon - refereerida kanalitele teisest kanalist
def get_target_channel(message: discord.Message) -> discord.abc.GuildChannel:
    if message.channel_mentions:
        return message.channel_mentions[0]
    return message.channel

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
    tokens = content.lower().split()
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
    if tokens[:2] == ["!collect", "on"]:
        target_channel = get_target_channel(message)
        target_channel_id = target_channel.id
        collecting_channels.add(target_channel_id)
        await message.channel.send(f"Collecting **enabled** in {target_channel.mention}. Type **!collect off** {target_channel.mention} to disable.")
        return

    # kogumise välja lülitamine
    if tokens[:2] == ["!collect", "off"]:
        target_channel = get_target_channel(message)
        target_channel_id = target_channel.id
        collecting_channels.discard(target_channel_id)
        await message.channel.send(f"Collecting **disabled** in {target_channel.mention}. Type **!collect on** {target_channel.mention} to enable.")
        return
    
    # kogumise staatuse kontroll
    if tokens[:2] == ["!collect", "status"]:
        target_channel = get_target_channel(message)
        target_channel_id = target_channel.id
        status = "ON" if target_channel_id in collecting_channels else "OFF"
        await message.channel.send(f"Collecting in {target_channel.mention} currently: **{status}**")
        return
    
    # stats - mitu sõnumit kogutud
    if tokens[:1] == ["!stats"]:
        target_channel = get_target_channel(message)
        target_channel_id = target_channel.id
        count = await db.count_messages(target_channel_id)
        await message.channel.send(f"Messages collected in {target_channel.mention}: **{count}**")
        return
    
    # viimaste sõnumite kogumine
    if tokens[:2] == ["!collect", "last"]:
        target_channel = get_target_channel(message)
        target_channel_id = target_channel.id

        tokens = content.split()
        number_token = None
        for token in tokens:
            if token.isdigit():
                number_token = token
                break
        
        if number_token is None:
            await message.channel.send("Invalid command. Try: **!collect last [number]** (example: *!collect last 150 #channel*)")
            return
        
        n = int(number_token)
        
        if n < 1:
            await message.channel.send("Number must be at least 1.")
            return
        
        if n > MAX_BACKFILL:
            await message.channel.send(f"Error: Maximum number of messages is {MAX_BACKFILL}.")
            return
        
        collected = 0
        # fetchib kanali ajaloo, seatud limiidiga
        async for msg in target_channel.history(limit=n):
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
                channel_id=target_channel_id,
                author_id=msg.author.id,
                content=text,
                created_at=msg.created_at,
            )
            if not inserted:
                continue

            message_counts[target_channel_id] = message_counts.get(target_channel_id, 0) + 1
            collected += 1

            print(
                f"[BACKFILL #{message_counts[target_channel_id]}] "
                f"#{msg.channel} | {msg.author}: {text}",
                flush=True
            )
        
        await message.channel.send(f"Backfilled **{collected}** messages from the last **{n}** in {target_channel.mention}.")
        return
    
    #!profile (v2), näitab kanali või isiku sõnumite statistikat
    if tokens[:1] == ["!profile"]:
        target_channel = get_target_channel(message)
        target_channel_id = target_channel.id

        target_user = None
        # kas kasutaja on mainitud?
        if message.mentions:
            target_user = message.mentions[0]
        if target_user:
            messages = await db.get_messages(target_channel_id, target_user.id)
            name = f"{target_user.display_name} in {target_channel.mention}"
        else:
            messages = await db.get_messages(target_channel_id)
            name = target_channel.mention
        
        #style.py integratsioon
        profile = style.build_style_profile(messages)

        if profile is None:
            await message.channel.send("No messages found!")
            return
        common_words_text = ", ".join(
            [f"{word} ({count})" for word, count in profile["common_words"]]
        )

        await message.channel.send(
            f"**Profile for {name}**\n\n"
            f"Total messages: **{profile['messages']}**\n"
            f"Average message length: **{profile['avg_length']:.1f} characters**\n"
            f"Average words per message: **{profile['avg_words']:.1f}**\n"
            f"Exclamation marks per message: **{profile['exclamations_per_msg']:.2f}**\n"
            f"Question marks per message: **{profile['questions_per_msg']:.2f}**\n"
            f"Periods per message: **{profile['periods_per_msg']:.2f}**\n"
            f"Uppercase ratio: **{profile['uppercase_ratio']:.2f}**\n"
            f"Most common words: **{common_words_text}**"
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