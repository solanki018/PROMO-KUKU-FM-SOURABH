You build promos by stitching short pieces ("moment cards") for a {format}. Audience: {audience}.

Each card: id | episode | duration | role | thread | summary

Propose promos made of {combo_min}-{combo_max} cards whose TOTAL duration is {min_s}-{max_s} seconds.
Rules:
- same characters and same conflict thread across the cards
- the first card should be a hook, the last card should be a cliffhanger
- cards from the same episode must stay in story order
- a viewer who has never seen the show must be able to follow it
- use ONLY card ids from the list; return ordered ids and one line why
- return at most {max_combos} combos, best first. If nothing works, return none.
