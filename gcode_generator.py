import csv
import re
import math
import numpy as np

# Global variables
stmm = 80      # steps per millimeter
posx = 100     # starting X position
posy = 100     # starting Y position

SEMITONES = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}

NOTE_RE = re.compile(r"^([A-G])(#?)([0-9])$")


def note_to_number(note: str) -> int:
    """
    Converts note names to indexes.

    C0  -> 0
    C#0 -> 1
    ...
    B9  -> 119
    P   -> -1
    """

    note = note.strip().upper()

    if note == "P":
        return -1

    match = NOTE_RE.fullmatch(note)
    if not match:
        raise ValueError(f"Invalid note format: {note}")

    base_note = match.group(1)
    sharp = match.group(2)
    octave = int(match.group(3))

    note_name = base_note + sharp

    if note_name not in SEMITONES:
        raise ValueError(f"Invalid note name: {note_name}")

    return octave * 12 + SEMITONES[note_name]


def load_frequency_table(filename: str) -> np.ndarray:
    """
    Reads the frequency CSV.

    Expected format:

    Note_index,Note,Octave,Frequency
    0,C,0,16.351
    1,C#,0,17.324
    ...

    Returns:
        frequency_by_index[index] = frequency
    """

    frequency_by_index = np.full(120, np.nan, dtype=np.float32)

    with open(filename, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            note_index = int(row["Note_index"])
            frequency = float(row["Frequency"])

            if not 0 <= note_index <= 119:
                raise ValueError(f"Note index out of range: {note_index}")

            frequency_by_index[note_index] = frequency

    if np.isnan(frequency_by_index).any():
        missing = np.where(np.isnan(frequency_by_index))[0]
        raise ValueError(f"Missing frequencies for note indexes: {missing}")

    return frequency_by_index


def load_note_csv(note_filename: str, frequency_filename: str):
    """
    Reads a melody CSV with rows like:

    C1,500
    D#2,250
    P,1000

    Returns:
        pitches:     note indexes, pause = -1
        durations:   durations in milliseconds
        frequencies: frequency for each row, pause = 0.0
        feedrates:   frequency * 60 / stmm
    """

    frequency_by_index = load_frequency_table(frequency_filename)

    pitches = []
    durations = []
    frequencies = []
    feedrates = []

    with open(note_filename, newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)

        for line_number, row in enumerate(reader, start=1):
            if not row or len(row) < 2:
                continue

            note_text = row[0].strip()
            duration_text = row[1].strip()

            try:
                duration = int(duration_text)
            except ValueError:
                # Allows a header row like: note,duration
                if line_number == 1:
                    continue
                raise ValueError(
                    f"Invalid duration on line {line_number}: {duration_text}"
                )

            if duration <= 0:
                raise ValueError(
                    f"Duration must be positive on line {line_number}: {duration}"
                )

            pitch = note_to_number(note_text)

            if pitch == -1:
                frequency = 0.0
            else:
                frequency = frequency_by_index[pitch]

            feedrate = frequency * 60 / stmm

            pitches.append(pitch)
            durations.append(duration)
            frequencies.append(frequency)
            feedrates.append(feedrate)

    return (
        np.array(pitches, dtype=np.int16),
        np.array(durations, dtype=np.int32),
        np.array(frequencies, dtype=np.float32),
        np.array(feedrates, dtype=np.float32),
    )


def analyse_file(note_filename: str, frequency_filename: str):
    pitches, durations, frequencies, feedrates = load_note_csv(
        note_filename,
        frequency_filename
    )

    total_duration = durations.sum()
    pause_duration = durations[pitches == -1].sum()
    sound_duration = durations[pitches != -1].sum()

    print(f"File: {note_filename}")
    print(f"Rows: {len(pitches)}")
    print(f"Total duration: {total_duration} ms")
    print(f"Sound duration: {sound_duration} ms")
    print(f"Pause duration: {pause_duration} ms")

    print("First 10 rows:")
    for i in range(min(10, len(pitches))):
        print(
            f"pitch={pitches[i]}, "
            f"duration={durations[i]} ms, "
            f"frequency={frequencies[i]} Hz, "
            f"feedrate={feedrates[i]} mm/min"
        )

    print()

    return pitches, durations, frequencies, feedrates


def create_gcode(
    x_durations,
    x_feedrates,
    y_durations,
    y_feedrates,
    output_filename="music.gcode"
):
    """
    Creates G-code where:
        - X motor plays melody 1
        - Y motor plays melody 2

    Durations are in milliseconds.
    Feedrates are in mm/min.

    Distance formula:
        distance_mm = feedrate_mm_per_min * duration_ms / 60000

    Movement direction alternates after every real movement.
    Starting position is taken from global variables posx and posy.
    """

    current_x = float(posx)
    current_y = float(posy)

    x_index = 0
    y_index = 0

    x_remaining = int(x_durations[0]) if len(x_durations) > 0 else 0
    y_remaining = int(y_durations[0]) if len(y_durations) > 0 else 0

    # First movement goes in the positive direction.
    # Then it alternates: +, -, +, -, ...
    direction = 1

    gcode = []

    gcode.append("; Generated music G-code")
    gcode.append("; X axis = melody 1")
    gcode.append("; Y axis = melody 2")
    gcode.append("")
    gcode.append("G21 ; use millimeters")
    gcode.append("G90 ; absolute positioning")
    gcode.append(f"G0 X{current_x:.3f} Y{current_y:.3f} ; move to start position")
    gcode.append("")

    while x_index < len(x_durations) or y_index < len(y_durations):

        # Move to the next X note if the current one has finished.
        while x_index < len(x_durations) and x_remaining <= 0:
            x_index += 1
            if x_index < len(x_durations):
                x_remaining = int(x_durations[x_index])

        # Move to the next Y note if the current one has finished.
        while y_index < len(y_durations) and y_remaining <= 0:
            y_index += 1
            if y_index < len(y_durations):
                y_remaining = int(y_durations[y_index])

        # Stop when both melodies are finished.
        if x_index >= len(x_durations) and y_index >= len(y_durations):
            break

        active_remaining_times = []

        if x_index < len(x_durations):
            active_remaining_times.append(x_remaining)

        if y_index < len(y_durations):
            active_remaining_times.append(y_remaining)

        # Use the shortest remaining note duration.
        # This keeps both melodies synchronized even if their note lengths differ.
        segment_duration_ms = min(active_remaining_times)

        if segment_duration_ms <= 0:
            continue

        # Current feedrates for each axis.
        # If one melody has finished, that axis does not move.
        if x_index < len(x_feedrates):
            x_feedrate = float(x_feedrates[x_index])
        else:
            x_feedrate = 0.0

        if y_index < len(y_feedrates):
            y_feedrate = float(y_feedrates[y_index])
        else:
            y_feedrate = 0.0

        # Convert duration from milliseconds to minutes.
        segment_duration_min = segment_duration_ms / 60000.0

        # Distance each motor travels during this segment.
        x_distance = x_feedrate * segment_duration_min
        y_distance = y_feedrate * segment_duration_min

        new_x = current_x + direction * x_distance
        new_y = current_y + direction * y_distance

        movement_distance = math.sqrt(
            (new_x - current_x) ** 2 +
            (new_y - current_y) ** 2
        )

        if movement_distance == 0:
            # Both axes are paused.
            gcode.append(f"G4 P{segment_duration_ms} ; pause")
        else:
            # G-code G1 only has one F value.
            # For diagonal movement, this combined feedrate keeps
            # the X and Y components moving at their intended speeds.
            combined_feedrate = movement_distance / segment_duration_min

            gcode.append(
                f"G1 X{new_x:.3f} Y{new_y:.3f} F{combined_feedrate:.3f} "
                f"; {segment_duration_ms} ms"
            )

            current_x = new_x
            current_y = new_y

            # Next real movement goes in the opposite direction.
            direction *= -1

        # Reduce the remaining time for the current notes.
        if x_index < len(x_durations):
            x_remaining -= segment_duration_ms

        if y_index < len(y_durations):
            y_remaining -= segment_duration_ms

    gcode.append("")
    gcode.append("; End of music G-code")

    with open(output_filename, "w", encoding="utf-8") as file:
        file.write("\n".join(gcode))

    print(f"G-code written to {output_filename}")
    print(f"Final position: X{current_x:.3f}, Y{current_y:.3f}")

    return gcode


# Main program
file1_pitches, file1_durations, file1_frequencies, file1_feedrates = analyse_file(
    "SongChannel1.csv",
    "note freq.csv"
)

file2_pitches, file2_durations, file2_frequencies, file2_feedrates = analyse_file(
    "SongChannel2.csv",
    "note freq.csv"
)

gcode = create_gcode(
    file1_durations,
    file1_feedrates,
    file2_durations,
    file2_feedrates,
    output_filename="music.gcode"
)