Your philosophy is roughly:

Test real behaviour, not mocked choreography

You prefer tests that exercise the actual production logic and only replace the true external boundaries:

* databases
* filesystems
* network calls
* clocks
* queues
* hardware APIs
* third-party services
* browser/runtime boundaries

The thing under test should mostly be real. Its collaborators should be real where practical, or replaced with fakes/in-memory implementations/testcontainers, not elaborate mocks.

Dependency injection over monkeypatching

Your preferred seam is:

service = Service(repo=InMemoryRepo(), clock=FakeClock(...))

not:

monkeypatch.setattr(module, "repo", fake_repo)

You dislike brittle monkeypatching because it usually signals that the production code was not designed with clean seams. The fix should usually be better design: push side effects to the edges, pass dependencies explicitly, and keep core logic pure or close to pure.

Never patch the code under test

One of your harder rules is:

Do not patch the thing you are testing.

Patching a dependency at the module boundary is already a smell; patching internals of the unit under test is worse because it turns the test into an implementation lock-in exercise.

You want tests to prove:

* given this input/state,
* when this public behaviour runs,
* the observable outcome is correct.

Not:

* this private method was called,
* these exact internal steps happened,
* this mock received these arguments in this order,
* the code performed the choreography the test author imagined.

Prefer fakes and spies over mocks

You are not anti-test-double. You are anti-mock theatre.

Your hierarchy is probably:

1. Real implementation when cheap and deterministic.
2. In-memory fake when replacing an external boundary.
3. Testcontainer/local service when behaviour matters.
4. Spy when you need to observe that a boundary was used.
5. Mock only when the dependency is genuinely awkward, slow, nondeterministic, or external.

Mocks are acceptable as a narrow tool. They are not a design style.

Keep test concerns out of production code

You dislike:

* test-only branches
* test-only env vars
* production flags added purely to make testing easier
* special casing test mode
* hidden global knobs
* patching imports halfway through a test

The production code should expose honest seams through normal design, usually dependency inversion. Test-only behaviour belongs in test fixtures, fakes, builders, or test harnesses.

Deterministic, hermetic, parallel-safe

You tend to value tests that can run reliably in CI and in parallel:

* no shared global state
* no real wall-clock assumptions
* no shared temp paths/ports/schemas
* no dependency on test ordering
* no unbounded sleeps
* no ambient environment mutation unless carefully isolated

For Go, your rule of thumb was basically: if two go test invocations would conflict, t.Parallel() probably will too.

The deeper principle

Your testing philosophy is really an architecture philosophy:

Code should be naturally testable because its boundaries are explicit, its core behaviour is real and isolated, and its side effects are pushed to the edges.

So when a test needs heavy monkeypatching, you probably read that as evidence of one of these problems:

* hidden dependency
* global state
* hard-coded singleton/client/session
* side effects buried too deep
* poor module boundary
* implementation coupled to runtime/environment
* design optimised for convenience rather than maintainability

The ideal test gives real confidence, is boring to read, and would still pass after a sensible refactor.
