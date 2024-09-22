


Code
```Python
    ("abs",         lambda a: torch.abs(a),             [("a", rand((64, 64))),                                           ]),
    ("abs",         lambda a: torch.abs(a),             [("a", rand((128, 128))),                                           ]),
    ("add",         lambda a, b: a + b,                 [("a", rand((64, 64))),         ("b", rand((64, 64)))             ]),
    ("add",         lambda a, b: a + b,                 [("a", rand((128, 128))),       ("b", rand((128, 128)))         ]),
    ("linear",      lambda x, w, b: torch.nn.functional.linear(x, w, b), 
        [("x", rand((64, 64))), ("w", rand((64, 64))), ("b", rand((64,)))                                                  ]),
    ("linear",      lambda x, w, b: torch.nn.functional.linear(x, w, b), 
        [("x", rand((128, 128))), ("w", rand((128, 128))), ("b", rand((128,)))                                                  ]),
    ("relu",        lambda x: torch.nn.functional.relu(x),
        [("x", rand((64, 64)))                                                                                            ]),
    ("relu",        lambda x: torch.nn.functional.relu(x),
        [("x", rand((128, 128)))                                                                                            ]),
    ("softmax",     lambda a: torch.nn.functional.softmax(a, dim=0),
        [("x", rand((64, 64))+1), ("0", None)                                                                             ]),
    ("softmax",     lambda a: torch.nn.functional.softmax(a, dim=1),
        [("x", rand((128, 128))+1), ("1", None)                                                                             ]),
    ("conv2d",      lambda x, w, b: torch.nn.functional.conv2d(x.permute((0, 3, 1, 2)), w.permute((3, 2, 0, 1)), b, stride=1, padding=0, dilation=1, groups=1).permute((0, 2, 3, 1)),
        [("x", rand((1, 16, 16, 16))), ("w", rand((3, 3, 16, 16))), ("b", rand((16, ))), 
         ("(size_t[]){1, 1}, (size_t[]){0, 0}, (size_t[]){1, 1}, 1", None)                                              ]),
    ("conv2d",      lambda x, w, b: torch.nn.functional.conv2d(x.permute((0, 3, 1, 2)), w.permute((3, 2, 0, 1)), b, stride=1, padding=1, dilation=1, groups=1).permute((0, 2, 3, 1)),
        [("x", rand((1, 16, 16, 16))), ("w", rand((3, 3, 16, 16))), ("b", rand((16, ))),
         ("(size_t[]){1, 1}, (size_t[]){1, 1}, (size_t[]){1, 1}, 1", None)                                              ]),
    # ("nchw_to_nhwc",  lambda x: x.permute((0, 2, 3, 1)),  [("x", rand((1, 2, 3, 3)))                                    ]),
    # ("nhwc_to_nchw",  lambda x: x.permute((0, 3, 1, 2)),  [("x", rand((1, 3, 3, 2)))                                    ]),
    ("conv2d",      lambda x, w, b: torch.nn.functional.conv2d(x.permute((0, 3, 1, 2)), w.permute((3, 2, 0, 1)), b, stride=1, padding=1, dilation=1, groups=16).permute((0, 2, 3, 1)),
        [("x", rand((1, 16, 16, 256))), ("w", rand((3, 3, 16, 16))), ("b", rand((16, ))),
         ("(size_t[]){1, 1}, (size_t[]){1, 1}, (size_t[]){1, 1}, 16", None)                                             ]),
    
    ("layer_norm",   lambda x, w, b: torch.nn.functional.layer_norm(x, (x.shape[1], ), w, b, eps=1e-05), 
        [("x", rand((64, 64))), ("1", None), ("w", rand((64))), ("b", torch.zeros((64))), ("1e-05", None)         ]),
    ("layer_norm",   lambda x, w, b: torch.nn.functional.layer_norm(x, (x.shape[1], ), w, b, eps=1e-05), 
        [("x", rand((128, 128))), ("1", None), ("w", rand((128))), ("b", rand((128))), ("1e-05", None)                      ]),

```



RISCV CPU
```
abs:                    PASS  (28710 cycles)
abs:                    PASS  (114726 cycles)
add:                    PASS  (36915 cycles)
add:                    PASS  (147507 cycles)
linear:                 PASS  (2327412 cycles)
linear:                 PASS  (17696372 cycles)
relu:                   PASS  (36923 cycles)
relu:                   PASS  (147363 cycles)
softmax:                PASS  (499842 cycles)
softmax:                PASS  (1995918 cycles)
conv2d:                 PASS  (7386689 cycles)
conv2d:                 PASS  (9188865 cycles)
conv2d:                 [ERROR] Unsupported conv2d operation for groups other than 1 or in_channels
FAIL  (71605 cycles)
layer_norm:             PASS  (237313 cycles)
layer_norm:             PASS  (933313 cycles)
```

RISCV Vector
```
abs:                    PASS  (2341 cycles)
abs:                    PASS  (9253 cycles)
add:                    PASS  (2866 cycles)
add:                    PASS  (11314 cycles)
linear:                 PASS  (394100 cycles)
linear:                 PASS  (2164340 cycles)
relu:                   PASS  (2593 cycles)
relu:                   PASS  (10273 cycles)
softmax:                PASS  (499842 cycles)
softmax:                PASS  (1995918 cycles)
conv2d:                 PASS  (7386689 cycles)
conv2d:                 PASS  (9188865 cycles)
conv2d:                 [ERROR] Unsupported conv2d operation for groups other than 1 or in_channels
FAIL  (71605 cycles)
layer_norm:             PASS  (90561 cycles)
layer_norm:             PASS  (346945 cycles)
```


