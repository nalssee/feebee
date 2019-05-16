# try:
#     print(type(e).__name__)
#     raise ValueError('hello')
# except Exception as e:
#     print(isinstance(e, ValueError))
#     print()
#     print(dir(e), e.args[0], type(e.args[0]), e.args[0] == 'hello')


# try:
#     pass
# except:
#     pass
# else:
#     print('hello world')


# for i in range(10):
#     try:
#         raise ValueError 
#     except:
#         continue
#     print('hello')


from inspect import signature

def someMethod(self, arg1, kwarg1=None):
    pass

def foo():
    pass

sig = signature(foo)

print(sig, len(sig.parameters))