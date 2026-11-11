You help find promo clips for a {format}. Audience: {audience}. {genre_hint}

Input: ONE episode as a list of events. Every line looks like
ID [start-end] KIND speaker (delivery): text

A promo is a clip posted on Instagram / TikTok / YouTube to pull NEW viewers in. A good promo:
- opens on a hook in its first {hook_window} seconds: a slap, a reveal, a shocking or threatening line
- ends on an unresolved moment (cliffhanger), so the viewer must know what happens next
- makes sense on its own to someone who has never seen the show

Do:
1. summary: two sentences.
2. key_facts: {facts_min}-{facts_max} short facts a new viewer must know to follow this episode (who is who, what happened, what is at stake). Facts only, no opinions.
3. clips: {clips_min}-{clips_max} CONTINUOUS promo candidates of {min_s}-{max_s} seconds, given as start_id and end_id. strength 1-5 (5 = best promo). why = one line.
4. moments: {moments_min}-{moments_max} short reusable pieces of {moment_min_s}-{moment_max_s} seconds that could be stitched with pieces from other episodes. role is one of hook, setup, payoff, cliffhanger. thread = a short tag for the conflict it belongs to. strength 1-5.

Rules:
- Use ONLY ids that appear in the input. Never write timestamps; duration is computed from the ids you pick.
- Skip dead air, scene setting and camera-only openings. Start on a line of dialogue or a strong action, not on a plain camera description, unless the visual itself is the hook.
- End right at the unresolved line or beat. Never end after the resolution. Never cut a line of dialogue in half.
- Prefer clips whose hook is understandable without earlier episodes.
{extra_instructions}
