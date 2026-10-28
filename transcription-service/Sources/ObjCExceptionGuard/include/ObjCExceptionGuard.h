// A boundary for Objective-C exceptions, which Swift cannot catch.
//
// AVFoundation reports some malformed media by raising an NSException rather
// than returning an error. Uncaught, that aborts the whole service, and every
// queued request with it. The decoder runs its synchronous AVFoundation calls
// through `ObjCExceptionGuard.run` so a raised exception becomes a thrown
// Swift error for that one request.

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// The `NSError` domain of a caught exception. `userInfo` carries the
/// exception's `name` and `reason` under the keys below.
extern NSErrorDomain const ObjCExceptionGuardErrorDomain;
extern NSString *const ObjCExceptionGuardNameKey;
extern NSString *const ObjCExceptionGuardReasonKey;

@interface ObjCExceptionGuard : NSObject

/// Run `block`; if it raises an Objective-C exception, return NO and describe
/// it in `error`. Imported into Swift as `try ObjCExceptionGuard.run { … }`.
///
/// Swift frames inside `block` are unwound without running their cleanups, so
/// `block` should only call into Objective-C and assign plain results.
+ (BOOL)runBlock:(NS_NOESCAPE void (^)(void))block
           error:(NSError *_Nullable *_Nullable)error NS_SWIFT_NAME(run(_:));

@end

NS_ASSUME_NONNULL_END
