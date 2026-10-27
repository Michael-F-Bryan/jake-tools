import FluidAudio
import Testing

@testable import TranscriptionService

@Suite struct ReconcileTests {
    private func word(_ text: String, _ start: Double, _ end: Double, confidence: Double = 0.9) -> TimedWord {
        TimedWord(text: text, start: start, end: end, confidence: confidence)
    }

    @Test func wordsTakeTheSpeakerOfTheTurnOverlappingThemMost() {
        let result = Reconcile.reconcile(
            words: [word("hello", 0.0, 0.5), word("there", 0.6, 1.0), word("hi", 2.0, 2.4)],
            turns: [SpeakerTurn(speaker: "7", start: 0, end: 1.2), SpeakerTurn(speaker: "3", start: 1.9, end: 3)])

        #expect(result.segments.map(\.speaker) == ["speaker_1", "speaker_2"])
        #expect(result.segments.map(\.text) == ["hello there", "hi"])
        #expect(result.speakers.map(\.id) == ["speaker_1", "speaker_2"])
        #expect(result.speakers.map(\.speakingSeconds) == [1.2, 1.1])
        #expect(result.speakers.map(\.segmentCount) == [1, 1])
        #expect(result.unassignedWordCount == 0)
        #expect(result.segments.allSatisfy { $0.words.allSatisfy { !$0.overlap } })
    }

    @Test func speakerLabelsFollowFirstAppearanceNotEngineIds() {
        let result = Reconcile.reconcile(
            words: [word("a", 0, 1), word("b", 5, 6)],
            turns: [SpeakerTurn(speaker: "zeta", start: 4, end: 7), SpeakerTurn(speaker: "alpha", start: 0, end: 2)])

        #expect(result.segments.map(\.speaker) == ["speaker_1", "speaker_2"])
        #expect(result.segments[0].text == "a")
    }

    @Test func overlappingTurnsFlagTheWordAndPickTheLargerOverlap() {
        let result = Reconcile.reconcile(
            words: [word("both", 1.0, 2.0)],
            turns: [SpeakerTurn(speaker: "a", start: 0, end: 1.3), SpeakerTurn(speaker: "b", start: 1.2, end: 3)])

        let flagged = result.segments[0].words[0]
        #expect(flagged.speaker == "speaker_2")
        #expect(flagged.overlap)
    }

    @Test func wordsNearATurnSnapToItAndDistantWordsStayUnassigned() {
        let result = Reconcile.reconcile(
            words: [word("near", 2.1, 2.4), word("far", 5.0, 5.5)],
            turns: [SpeakerTurn(speaker: "a", start: 0, end: 2.0)])

        #expect(result.segments.count == 2)
        #expect(result.segments[0].speaker == "speaker_1")
        #expect(result.segments[1].speaker == nil)
        #expect(result.segments[1].words[0].speaker == nil)
        #expect(result.unassignedWordCount == 1)
    }

    @Test func longPausesSplitSegmentsEvenForOneSpeaker() {
        let result = Reconcile.reconcile(
            words: [word("one", 0, 0.5), word("two", 0.6, 1.0), word("three", 4.0, 4.5)],
            turns: [SpeakerTurn(speaker: "a", start: 0, end: 5)])

        #expect(result.segments.map(\.text) == ["one two", "three"])
        #expect(result.speakers[0].segmentCount == 2)
    }

    @Test func noTurnsLeavesEverythingUnassignedButStillSegmented() {
        let result = Reconcile.reconcile(words: [word("x", 0, 1), word("y", 1, 2)], turns: [])

        #expect(result.speakers.isEmpty)
        #expect(result.segments.count == 1)
        #expect(result.segments[0].speaker == nil)
        #expect(result.unassignedWordCount == 2)
    }

    @Test func tokensGroupIntoWordsOnTheSentencePieceBoundaryMarker() {
        let tokens = [
            TokenTiming(token: "▁Hel", tokenId: 1, startTime: 0.00, endTime: 0.08, confidence: 0.9),
            TokenTiming(token: "lo", tokenId: 2, startTime: 0.08, endTime: 0.16, confidence: 0.7),
            TokenTiming(token: ",", tokenId: 3, startTime: 0.16, endTime: 0.24, confidence: 1.0),
            TokenTiming(token: "▁world", tokenId: 4, startTime: 0.40, endTime: 0.56, confidence: 0.5),
            TokenTiming(token: "<blank>", tokenId: 0, startTime: 0.56, endTime: 0.64, confidence: 0.1),
        ]

        let words = Reconcile.words(fromTokens: tokens)

        #expect(words.map(\.text) == ["Hello,", "world"])
        #expect(words.map(\.start) == [0.0, 0.4])
        #expect(words.map(\.end) == [0.24, 0.56])
        #expect(abs(words[0].confidence - (0.9 + 0.7 + 1.0) / 3) < 1e-6)
    }

    @Test func wordWithoutAnyBoundaryMarkerStillStartsAWord() {
        let tokens = [
            TokenTiming(token: "ok", tokenId: 1, startTime: 0.00, endTime: 0.08, confidence: 0.9),
            TokenTiming(token: "ay", tokenId: 2, startTime: 0.08, endTime: 0.16, confidence: 0.9),
        ]
        #expect(Reconcile.words(fromTokens: tokens).map(\.text) == ["okay"])
    }
}
