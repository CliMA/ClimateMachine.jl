# Coding Conventions

A list of recommended coding conventions.

 - There are good recommendations in the Julia style-guide:
    - https://docs.julialang.org/en/v1/manual/style-guide/index.html
    - https://docs.julialang.org/en/v0.6/manual/packages/#Guidelines-for-naming-a-package-1

 - While Unicode is permissable in the Julia language, we recommend to avoid using it (see below for why).

 - Modules, and class names (structs), should follow TitleCase convention. Note that class names cannot coincide with module names.

 - Function names should be lowercase, with words separated by underscores as necessary to improve readability.

 - Variable names follow the same convention as function names. Follow CMIP conventions (http://clipc-services.ceda.ac.uk/dreq/) where possible and practicable.

 - Make names consistent, distinctive, and meaningful.

 - Document design and purpose, rather than mechanics and implementation (document interfaces and embed documentation in code).

 - Avoid variable names that coincide with module and class names, as well as function/variable names that are natively supported.

 - Never use the characters 'l' (lowercase letter el), 'O' (uppercase letter oh), or 'I' (uppercase letter eye) as single character variable names.

 - Two white spaces are used for indent. This is not part of the standard convention, but recent development efforts have been using this indentation style (e.g., Google's Tensorflow), and this style is being used here also.

 - KISS (keep it simple stupid).

 - Try to limit all lines to a maximum of 79 characters.

 - Single access point - if a variable/constant is defined more than once, then move it into a module and import (or "using") to that module to access the variable in order to enforce a single access point (to avoid consistency issues).

 - 2 or more, use a "for" - If you are looping over indexes, and you loop over at least two, then just use a for loop. Example:

   - Do this
      `temp = [4, 2, 6]`
      `for i in 1:3`
        `  a[i] = temp[i]`
      `end`

   - Not this
      `a[1] = 4`
      `a[2] = 2`
      `a[3] = 6`

 - "import"/"using" should be grouped in the following order:

   - Standard library imports.
   - Related third party imports.
   - Local application/library specific imports.
   - Use a blank line between each group of imports.



## * Why no Unicode for now?

 - Some characters are visibly indistinguishable. Capital "a" and capital alpha are visibly indistinguishable, but are recognized as separate characters (e.g., search distinguishable).

 - Some characters are difficult to read. Sometimes, the overline/overdot/hats overlap with characters making them difficult to see.

 - Portability issues. Unicode does not render in Jupyter notebook natively (on OSX).

 - If it does improve readability enough, and are not worried about portability, we may introduce a list of permissable characters that are commonly used.


