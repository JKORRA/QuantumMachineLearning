"""Data processing modules for MIDI parsing and encoding."""

from .midi_parser import MidiParser, extract_notes_from_midi, export_to_midi, Note
from .encoder import PitchEncoder, PitchVelocityEncoder
from .datasets import SimpleDataset, ComplexDataset, create_target_distribution
from .duration_encoder import DurationEncoder, PitchDurationEncoder, DURATION_BINS

__all__ = [
    'MidiParser',
    'extract_notes_from_midi',
    'export_to_midi',
    'Note',
    'PitchEncoder',
    'PitchVelocityEncoder',
    'DurationEncoder',
    'PitchDurationEncoder',
    'DURATION_BINS',
    'SimpleDataset',
    'ComplexDataset',
    'create_target_distribution',
]
