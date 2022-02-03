# Computational design
- partially denormalized - OLAP not OLTP, optimized for read performance
- 1st normal form - single value / cell, same data type for every column
- 2nd normal form - every column depends on pk
- 3rd normal form - dependencies are put in seperate tables
- 4th normal form - dupblicate data put in separate tables


# Philosophy
- natural language / stories have eloquent *imagined* content / ideas, our brain does part of that translation into visual content
- creating a translation of meaning into visual form is a step forward in investigating the mechanics of the human mind
- how do people translate language into visuals? text has *connotative meaning*, imagery with which it is associated
  - e.g. the end of the world is associated in mainstream media with apocalyptic imager (fires, zombies, etc.)
- beyond content, text also has connotative composition / scale
  - e.g. end of the world has a large scale- but it's not just because "the world" is large, it's also because "the end" is a broad idea. another example would be "life" or "the meaning of life", neither of which are large in scale, but both of which evoke visual scale
- the exact mechanisms of this language-composition relationship are not clear, but we can simulate them with FSL/FT on GPT3 or smaller models


# Design

- Evaluation: trinary scale (-1, 0, 1)
  - Bad, 0, Good
  - Boring, 0, Novel
  - Ugly, 0, Aesthetic
  - Meaningless, 0, Meaningful
- 