"""
MIDI Parser Module
==================

Parses MIDI files and extracts musical features (pitch, velocity, duration).
"""

import mido
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path


@dataclass
class Note:
    """Represents a single musical note."""
    pitch: int          # MIDI pitch (0-127)
    velocity: int       # MIDI velocity (0-127)
    start_time: float   # Start time in ticks
    duration: float     # Duration in ticks
    
    @property
    def note_name(self) -> str:
        """Convert MIDI pitch to note name (e.g., 'C4')."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = self.pitch // 12 - 1
        return f"{note_names[self.pitch % 12]}{octave}"


class MidiParser:
    """
    Parser for MIDI files.
    
    Extracts notes with pitch, velocity, timing, and duration information.
    
    Example
    -------
    >>> parser = MidiParser('mario.mid')
    >>> notes = parser.parse()
    >>> print(f"Found {len(notes)} notes")
    >>> pitches = parser.get_pitches()
    """
    
    def __init__(self, midi_path: str | Path):
        """
        Initialize the MIDI parser.
        
        Parameters
        ----------
        midi_path : str or Path
            Path to the MIDI file.
        """
        self.midi_path = Path(midi_path)
        if not self.midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {self.midi_path}")
        
        self.midi = mido.MidiFile(str(self.midi_path))
        self.notes: List[Note] = []
        self._parsed = False
    
    @property
    def ticks_per_beat(self) -> int:
        """Ticks per beat (quarter note)."""
        return self.midi.ticks_per_beat
    
    @property
    def duration_seconds(self) -> float:
        """Total duration in seconds."""
        return self.midi.length
    
    @property
    def n_tracks(self) -> int:
        """Number of tracks in the MIDI file."""
        return len(self.midi.tracks)
    
    def parse(self, track_indices: Optional[List[int]] = None) -> List[Note]:
        """
        Parse the MIDI file and extract notes.
        
        Parameters
        ----------
        track_indices : list of int, optional
            Indices of tracks to parse. If None, parse all tracks.
        
        Returns
        -------
        list of Note
            Extracted notes from the MIDI file.
        """
        self.notes = []
        
        if track_indices is None:
            track_indices = list(range(len(self.midi.tracks)))
        
        for track_idx in track_indices:
            if track_idx >= len(self.midi.tracks):
                continue
            
            track = self.midi.tracks[track_idx]
            self._parse_track(track)
        
        self._parsed = True
        return self.notes
    
    def _parse_track(self, track: mido.MidiTrack) -> None:
        """Parse a single track and extract notes."""
        active_notes = {}  # pitch -> (start_time, velocity)
        current_time = 0
        
        for msg in track:
            current_time += msg.time
            
            if msg.type == 'note_on' and msg.velocity > 0:
                # Note starts
                active_notes[msg.note] = (current_time, msg.velocity)
            
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                # Note ends
                if msg.note in active_notes:
                    start_time, velocity = active_notes.pop(msg.note)
                    duration = current_time - start_time
                    
                    note = Note(
                        pitch=msg.note,
                        velocity=velocity,
                        start_time=start_time,
                        duration=duration
                    )
                    self.notes.append(note)
    
    def get_pitches(self) -> np.ndarray:
        """Get array of all pitches."""
        if not self._parsed:
            self.parse()
        return np.array([note.pitch for note in self.notes])
    
    def get_velocities(self) -> np.ndarray:
        """Get array of all velocities."""
        if not self._parsed:
            self.parse()
        return np.array([note.velocity for note in self.notes])
    
    def get_durations(self) -> np.ndarray:
        """Get array of all durations."""
        if not self._parsed:
            self.parse()
        return np.array([note.duration for note in self.notes])
    
    def get_pitch_velocity_pairs(self) -> np.ndarray:
        """Get array of (pitch, velocity) pairs."""
        if not self._parsed:
            self.parse()
        return np.array([[note.pitch, note.velocity] for note in self.notes])
    
    def get_statistics(self) -> dict:
        """
        Compute statistics about the parsed MIDI.
        
        Returns
        -------
        dict
            Dictionary containing various statistics.
        """
        if not self._parsed:
            self.parse()
        
        pitches = self.get_pitches()
        velocities = self.get_velocities()
        durations = self.get_durations()
        
        return {
            'n_notes': len(self.notes),
            'n_unique_pitches': len(np.unique(pitches)),
            'pitch_range': (int(pitches.min()), int(pitches.max())),
            'pitch_mean': float(pitches.mean()),
            'pitch_std': float(pitches.std()),
            'n_unique_velocities': len(np.unique(velocities)),
            'velocity_range': (int(velocities.min()), int(velocities.max())),
            'velocity_mean': float(velocities.mean()),
            'velocity_std': float(velocities.std()),
            'duration_mean': float(durations.mean()),
            'duration_std': float(durations.std()),
        }


def extract_notes_from_midi(midi_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to extract pitches and velocities from a MIDI file.
    
    Parameters
    ----------
    midi_path : str or Path
        Path to the MIDI file.
    
    Returns
    -------
    pitches : ndarray
        Array of MIDI pitches.
    velocities : ndarray
        Array of MIDI velocities.
    """
    parser = MidiParser(midi_path)
    parser.parse()
    return parser.get_pitches(), parser.get_velocities()


def export_to_midi(
    notes: List[Note],
    output_path: str | Path,
    tempo: int = 120,
    ticks_per_beat: int = 480
) -> None:
    """
    Export a list of Note objects to a MIDI file.
    
    Parameters
    ----------
    notes : list of Note
        List of Note objects to export.
    output_path : str or Path
        Path where the MIDI file will be saved.
    tempo : int, default=120
        Tempo in beats per minute (BPM).
    ticks_per_beat : int, default=480
        MIDI ticks per beat (quarter note).
    
    Example
    -------
    >>> notes = [Note(pitch=60, velocity=80, start_time=0.0, duration=0.5)]
    >>> export_to_midi(notes, 'output.mid', tempo=120)
    """
    # Create a new MIDI file
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    
    # Set tempo
    microseconds_per_beat = mido.bpm2tempo(tempo)
    track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0))
    
    # Convert seconds to ticks
    seconds_per_tick = 60.0 / (tempo * ticks_per_beat)
    
    # Create note events
    events = []
    for note in notes:
        # Convert times from seconds to ticks
        start_tick = int(note.start_time / seconds_per_tick)
        end_tick = int((note.start_time + note.duration) / seconds_per_tick)
        
        # Ensure pitch and velocity are integers (MIDI requirement)
        pitch = int(round(note.pitch))
        velocity = int(round(note.velocity))
        
        # Clip to valid MIDI range
        pitch = max(0, min(127, pitch))
        velocity = max(0, min(127, velocity))
        
        # Add note_on and note_off events
        events.append((start_tick, 'note_on', pitch, velocity))
        events.append((end_tick, 'note_off', pitch, 0))
    
    # Sort events by time
    events.sort(key=lambda x: x[0])
    
    # Convert absolute times to delta times and add to track
    last_time = 0
    for abs_time, msg_type, pitch, velocity in events:
        delta_time = abs_time - last_time
        
        if msg_type == 'note_on':
            track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=delta_time))
        else:
            track.append(mido.Message('note_off', note=pitch, velocity=velocity, time=delta_time))
        
        last_time = abs_time
    
    # Save the MIDI file
    midi.save(str(output_path))
