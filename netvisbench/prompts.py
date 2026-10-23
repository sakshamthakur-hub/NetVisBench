"""Prompts for the four evaluation conditions.

* vision - the model sees the image and the question.
* text   - the model sees the same facts written as text, no image.
* blind  - the model sees only the question (measures guessing / priors).
* sgd    - Schema-Guided Describe-then-Reason (our method): call 1 turns
           the image into a structured transcription using a short
           family-specific schema; call 2 answers from that transcription
           only. Perception and reasoning are separated.
"""

SYSTEM = ("You are an expert in computer networking. Work through the problem carefully and "
          "briefly. End your reply with a final line of the form 'ANSWER: <answer>' and nothing "
          "after it.")

TRANSCRIBE_SYSTEM = ("You are a meticulous transcriber of computer-networking figures. "
                     "You copy every label and value exactly as printed. You do not solve "
                     "anything.")

# What to extract from each kind of figure. Generic to the figure type:
# none of these hints reveal anything about a specific question.
SCHEMAS = {
    "topology": ("List every router name. Then list EVERY link as 'X-Y: cost', one per line. "
                 "Check each router in turn and make sure all links touching it are listed."),
    "subnet": ("For each LAN, write its name and subnet (CIDR) and then every host on it as "
               "'Hn=IP address'. Copy the numbers digit by digit."),
    "tcp_seq": ("List the client's data segments from top to bottom as 'Seq=..., Len=...', and "
                "say for each whether it reaches the server or is lost (red X). Then list the "
                "server's ACK labels from top to bottom. Note any retransmission."),
    "cwnd": ("Read the plotted curve point by point and write the congestion window value at "
             "every transmission round, as 'round: value', for ALL rounds shown."),
    "ipv4_header": ("Write every field of the header as 'Field: value', copying the value exactly "
                    "as printed (keep hexadecimal or binary notation as shown)."),
    "capture": ("Transcribe the packet list row by row as 'No | Source | Destination | Info', "
                "copying ports, flags, Seq, Ack and Len exactly. Include every row."),
}


def vision_prompt(question: str) -> str:
    return f"{question}\n\nThe figure is attached."


def text_prompt(question: str, text_rendering: str) -> str:
    q = question.replace("The diagram shows", "The following description shows") \
                .replace("The diagram is", "The following description is") \
                .replace("The plot shows", "The following data shows") \
                .replace("The image shows", "The following listing shows")
    return f"{text_rendering}\n\n{q}"


def blind_prompt(question: str) -> str:
    return (f"{question}\n\n(The figure is not available to you. Give your best guess anyway; "
            f"you must still end with an ANSWER line.)")


def transcribe_prompt(family: str) -> str:
    return ("Transcribe the attached computer-networking figure into structured text so that "
            "someone who cannot see it has all of its information.\n" + SCHEMAS[family] +
            "\nOutput only the transcription.")


def answer_from_transcript_prompt(question: str, transcript: str) -> str:
    return ("Below is a transcription of a computer-networking figure, followed by a question "
            "about the figure. Answer using the transcription.\n\n=== TRANSCRIPTION ===\n"
            f"{transcript}\n=== END ===\n\n{question}")
