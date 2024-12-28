
# Switching to Header-only Library

A decision is made to switch to a header-only library, since Baremetal-NN is designed to be a small and self-contained library, with only external dependencies on the standard library. The user can copy the `nn.h` file to their project and include it once in their application code where the neural network-related code is defined and used. 


## When to use a header-only library?

### Pros

#### 1. Ease of Use

- Single File Inclusion: Users only need to include a single header file to use the library. This simplifies the setup process and eliminates the need to link against compiled binaries.
- No Separate Compilation: No need to compile the library separately or manage its object files.

#### 2. Inlined Code Optimization

- Better Inlining: Function definitions in headers can be inlined by the compiler, potentially improving performance by eliminating function call overhead.
- Optimization Scope: Since the entire code is visible at the inclusion point, the compiler can apply optimizations across the entire translation unit.

#### 3. Portability

- No Dependencies on Precompiled Binaries: The library can be easily used across different platforms without worrying about binary compatibility.
- Source Distribution: Users always get the source code, making it easier to debug, customize, or audit the library.

#### 4. Simpler Distribution

- The library is distributed as a single file or a few files, making it easier to package, share, or include in projects.

#### 5. No Linker Issues

- Avoids linker errors since the code is compiled directly into each translation unit where it is included.

### Cons

#### 1. Code Duplication

- Increased Binary Size: Each translation unit that includes the header will have its own copy of the library's code, potentially bloating the final binary size.
- Increased Compilation Time: The same code may be compiled multiple times for different translation units, leading to longer build times.

#### 2. Debugging Complexity

- Inlined Functions: Debugging inlined functions can be harder because they may not appear clearly in stack traces or debugging tools.
- Compiler-Specific Behavior: Different compilers may handle the inlined code differently, leading to inconsistencies.

#### 3. Maintenance Challenges

- Header Dependency Hell: Header files can include other headers, leading to complex dependency chains and potential circular dependencies.
- Limited Encapsulation: All implementation details are exposed in the header, which can clutter the global namespace and make it harder to enforce encapsulation.

#### 4. Binary Compatibility Issues

- ABI Changes: Any change to the library (e.g., modifying a function signature) will require recompiling all dependent code, as the library is directly included in the user's codebase.

#### 5. Conditional Compilation Complexity

- Preprocessor Directives: Heavy use of #ifdef and #define for managing configuration or conditional compilation can make the code harder to read and maintain.

## When to Use a Header-Only Library

A header-only library is ideal when:

- The library is small and self-contained.
- The library heavily uses templates or inline functions, making it impractical to separate interface and implementation.
- Portability and ease of distribution are priorities.
- Performance gains from inlining outweigh the drawbacks of code duplication.

## When to Avoid a Header-Only Library

A header-only library might not be the best choice when:

- The library is large, leading to significant code duplication and increased compile times.
- Binary size is a critical concern.
- You want to enforce a clear separation between interface and implementation.
- Maintaining strict ABI stability is essential.

## Conclusion

Header-only libraries are a trade-off between convenience and efficiency. They shine in situations requiring simplicity and portability but might not be suitable for large, complex libraries where modularization and compilation efficiency are more critical.
