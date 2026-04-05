# Parengus Utils

This is intended to be an extension to the stdlib like EASTL. Unlike EASTL this
is not meant to be a full drop in replacement but provide convenient wrappers,
helpful policy options such as using non-randomized hash tables, and bridge to
common crates that imo should be part of the standard library, such as
deterministic floating point types via libm (useful for cross machine
simulation), non-poisoning mutexes via parking lot, random numbers via `rand`,
and other useful utilities. Also it renames BTreeMap to OrderedMap because
naming is important

## Inclusion Policy

***THIS IS NOT A DUMPING GROUND!*** This is explicitly a lean library designed
to be a supplement. Hyperspecific data structures such as Tries do not go here.
Presented are a few examples of datastructures that have been considered for
this crate and been rejected, as well as reasoning

- Tombstone(Hash|Ordered)Map: A map in which the initial delete does not delete
  the value but instead leaves behind a tombstone value.
  - Example use case: In winit, Windows are easily destroyed by dropping them.
    However, after dropping them, you will recieve a WindowEvent::Destroyed with
    that window id. Other events may also appear relating to that WindowId, I'm
    not sure. By leaving a tombstone behind, we can use the WindowId as a key to
    an internal Window struct, deleting the Window struct will leave a tombstone
    value letting us know to no-op on those events instead of panicking because
    winit handed us a WindowId we don't know about indicating something has gone
    wrong.
  - Why it is excluded: Realistically this can be simulated with just keeping an
    Option in the value type. That doesn't mean this doesn't want to be here,
    but it's a lot of logic for something that has a low weight solution.
- InsertionOrderedHashMap: A hashmap that remembers its insertion order, using
  that to provide stable iteration
  - This is a type level encoding of a pattern shown in the (second swisstable
    cppcon talk)[https://youtu.be/JZE3_0qvrMg?si=3tDbR6HA0k_Z3JPY&t=8574]. The
    motivating concept is that we have a service with O(N) inputs but we have
    randomizing hashtables. This means we now have O(N!) possible outputs. This
    defeats caching and other optimizations. Instead we keep the insertion order
    by inserting the key into a vector as well as the table. Then we can compose
    iterators such that we iterate through the hashtable using the keys in the
    vector. Very convenient.
  - Why this is excluded: Unfortunately this is overly simplistic and there are
    a lot of devils in the details. For example, what happens when you insert
    the same key multiple times? There are multiple different potential answers,
    ommited for brevity, and there are tradeoffs for each. Implementing a way to
    be generic over that policy is hypothetically possible, but not reasonable.