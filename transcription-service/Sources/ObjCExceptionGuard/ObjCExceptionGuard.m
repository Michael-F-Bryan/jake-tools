#import "ObjCExceptionGuard.h"

NSErrorDomain const ObjCExceptionGuardErrorDomain = @"ObjCExceptionGuard";
NSString *const ObjCExceptionGuardNameKey = @"ExceptionName";
NSString *const ObjCExceptionGuardReasonKey = @"ExceptionReason";

@implementation ObjCExceptionGuard

+ (BOOL)runBlock:(NS_NOESCAPE void (^)(void))block error:(NSError *_Nullable *_Nullable)error {
    @try {
        block();
        return YES;
    } @catch (NSException *exception) {
        if (error) {
            *error = [NSError errorWithDomain:ObjCExceptionGuardErrorDomain
                                         code:1
                                     userInfo:@{
                                         ObjCExceptionGuardNameKey: exception.name ?: @"",
                                         ObjCExceptionGuardReasonKey: exception.reason ?: @"",
                                         NSLocalizedDescriptionKey: [NSString
                                             stringWithFormat:@"Objective-C exception %@: %@",
                                                              exception.name, exception.reason ?: @""],
                                     }];
        }
        return NO;
    }
}

@end
