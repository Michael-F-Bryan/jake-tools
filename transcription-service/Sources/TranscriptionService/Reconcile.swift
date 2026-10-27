// Reconciling timed words with diarised speaker turns. Pure functions over
// plain values, so the rules are testable without a model.
//
// Rules:
// - Speakers are renamed `speaker_1`, `speaker_2`, … in order of first
//   appearance in the diarisation, request-wide. They are anonymous labels.
// - A word goes to the turn that overlaps it the most. If turns of more than
//   one speaker overlap it, it is flagged `overlap`.
// - A word no turn overlaps snaps to the nearest turn within
//   `maxSnapGapSeconds`; otherwise it stays unassigned (`speaker: null`).
// - Segments are maximal runs of consecutive words with the same speaker,
//   also split at pauses longer than `maxSegmentGapSeconds`.

import FluidAudio
import Foundation

public struct TimedWord: Sendable, Equatable {
    public var text: String
    public var start: Double
    public var end: Double
    public var confidence: Double

    public init(text: String, start: Double, end: Double, confidence: Double) {
        self.text = text
        self.start = start
        self.end = end
        self.confidence = confidence
    }
}

public struct SpeakerTurn: Sendable, Equatable {
    /// The engine's own label; only used to tell speakers apart.
    public var speaker: String
    public var start: Double
    public var end: Double

    public init(speaker: String, start: Double, end: Double) {
        self.speaker = speaker
        self.start = start
        self.end = end
    }
}

public struct Reconciliation: Sendable, Equatable {
    public var segments: [Segment]
    public var speakers: [SpeakerSummary]
    public var unassignedWordCount: Int
}

public enum Reconcile {
    public static let defaultMaxSnapGapSeconds = 0.5
    public static let defaultMaxSegmentGapSeconds = 2.0

    public static func reconcile(
        words: [TimedWord],
        turns: [SpeakerTurn],
        maxSnapGapSeconds: Double = defaultMaxSnapGapSeconds,
        maxSegmentGapSeconds: Double = defaultMaxSegmentGapSeconds
    ) -> Reconciliation {
        let orderedTurns = turns.sorted { ($0.start, $0.end) < ($1.start, $1.end) }
        var labels: [String: String] = [:]
        var labelOrder: [String] = []
        for turn in orderedTurns where labels[turn.speaker] == nil {
            let label = "speaker_\(labels.count + 1)"
            labels[turn.speaker] = label
            labelOrder.append(label)
        }

        var assigned: [Word] = []
        var unassigned = 0
        for word in words.sorted(by: { ($0.start, $0.end) < ($1.start, $1.end) }) {
            let (speaker, overlap) = attribute(
                word, turns: orderedTurns, labels: labels, maxSnapGapSeconds: maxSnapGapSeconds)
            if speaker == nil { unassigned += 1 }
            assigned.append(
                Word(
                    text: word.text, startSeconds: word.start, endSeconds: word.end,
                    confidence: word.confidence, speaker: speaker, overlap: overlap))
        }

        var segments: [Segment] = []
        var run: [Word] = []
        func flush() {
            guard let first = run.first, let last = run.last else { return }
            segments.append(
                Segment(
                    speaker: first.speaker,
                    startSeconds: first.startSeconds,
                    endSeconds: last.endSeconds,
                    text: run.map(\.text).joined(separator: " "),
                    words: run))
            run.removeAll(keepingCapacity: true)
        }
        for word in assigned {
            if let previous = run.last,
                previous.speaker != word.speaker || word.startSeconds - previous.endSeconds > maxSegmentGapSeconds
            {
                flush()
            }
            run.append(word)
        }
        flush()

        var speaking: [String: Double] = [:]
        for turn in orderedTurns {
            speaking[labels[turn.speaker]!, default: 0] += max(0, turn.end - turn.start)
        }
        let speakers = labelOrder.map { label in
            SpeakerSummary(
                id: label,
                speakingSeconds: speaking[label] ?? 0,
                segmentCount: segments.filter { $0.speaker == label }.count)
        }
        return Reconciliation(segments: segments, speakers: speakers, unassignedWordCount: unassigned)
    }

    private static func attribute(
        _ word: TimedWord, turns: [SpeakerTurn], labels: [String: String], maxSnapGapSeconds: Double
    ) -> (speaker: String?, overlap: Bool) {
        var best: (label: String, overlap: Double)?
        var overlappingSpeakers: Set<String> = []
        var nearest: (label: String, gap: Double)?
        for turn in turns {
            let label = labels[turn.speaker]!
            let overlap = min(word.end, turn.end) - max(word.start, turn.start)
            if overlap > 0 {
                overlappingSpeakers.insert(label)
                if best == nil || overlap > best!.overlap {
                    best = (label, overlap)
                }
                continue
            }
            let gap = turn.end <= word.start ? word.start - turn.end : turn.start - word.end
            if nearest == nil || gap < nearest!.gap {
                nearest = (label, gap)
            }
        }
        if let best {
            return (best.label, overlappingSpeakers.count > 1)
        }
        if let nearest, nearest.gap <= maxSnapGapSeconds {
            return (nearest.label, false)
        }
        return (nil, false)
    }

    /// Group Parakeet's SentencePiece token timings into words: a token whose
    /// piece starts with the word-boundary marker begins a new word, every
    /// other token (including punctuation) continues the current one. A
    /// word's confidence is the mean of its tokens'.
    public static func words(fromTokens tokens: [TokenTiming]) -> [TimedWord] {
        var result: [TimedWord] = []
        var text = ""
        var start = 0.0
        var end = 0.0
        var confidences: [Double] = []

        func flush() {
            let trimmed = text.trimmingCharacters(in: .whitespaces)
            if !trimmed.isEmpty {
                let confidence = confidences.isEmpty ? 0 : confidences.reduce(0, +) / Double(confidences.count)
                result.append(TimedWord(text: trimmed, start: start, end: end, confidence: confidence))
            }
            text = ""
            confidences = []
        }

        for token in tokens {
            let piece = token.token
            if piece.isEmpty || piece == "<blank>" || piece == "<pad>" { continue }
            let startsWord = piece.hasPrefix("▁") || piece.hasPrefix(" ")
            if startsWord || text.isEmpty {
                flush()
                text = startsWord ? String(piece.dropFirst()) : piece
                start = token.startTime
            } else {
                text += piece
            }
            end = max(end, token.endTime)
            confidences.append(Double(token.confidence))
        }
        flush()
        return result
    }
}
