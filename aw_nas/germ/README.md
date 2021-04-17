## The "Germ" Subpackage

### Workflow

The `aw_nas.germ` subpackage supports a workflow as follows.
1. The developer writes a class definition that inherits `germ.GermSuperNet` in a code snippet. In the `__init__` method of the class, the developer declares various decisions (instances of `germ.BaseDecision` subclasses), and passes these decisions to instantiate various `SearchableBlocks` instances (e.g., `SearchableConvBNBlock` with variable `out_channels`, `kernel_size`, and `stride`). One example of the code snippet is in `examples/germ/example_code_snippet.py`.
2. **Germ** parses the snippet to generate a search space configuration file, which is a YAML file that would be loaded by `GermSearchSpace`.
3. Write a search configuration file that would be accepted by `awnas search`, and run the search process.



The `scripts/generate_germ_search_cfg.py` script helps do part of the work of the 2nd and 3rd step. Try run `python scripts/generate_germ_search_cfg.py examples/germ/example_code_snippet.py examples/germ/try_generate.yaml`, and you would see a generated __partial__ search configuration file `examples/germ/try_generate.yaml`. At this time, this file contains configurations of the **Germ** search space and **Germ** weights manager. Then, one should manually add the configurations of other components (i.e., dataset, trainer, evaluator, controller, objective) to this search configuration file. A complete search configuration using **Germ** components is in `examples/germ/example_code_snippet_search.yaml`. One could try run `awnas search examples/germ/example_code_snippet_search.yaml --gpu 0`.



A graphic illustration of this workflow would be soon available @TODO.



### Why Germ

**Ease of development** Previously, when one wants to run a NAS flow in a new search space, ze must write new definitions for both the search space and weights manager. In fact, these two components are tightly coupled, we'd like to reduce the developers' burden by only requiring them to write a supernet definition. With a placeholder-like declarative grammar, the developers can declare which choices are **searchable** in an intuitive way.

And then, based on the developer's supernet definition, the search space definition could be automatically extracted. This would ease the burden of development for new NAS applications. **Germ** aims to facilitate this type of workflow by unifying the declaration of searchable decisions via `BaseDecision` subclasses.

**General handling of decisions** Besides, compared with previous workflow in which a brandly new search space class is written for a new search space, the newly-introduced `Decision` abstraction makes it easy to develop general controllers (i.e., search strategies): The search logic developed for different `Decision`s can be combined to enable search in various search spaces declared with **Germ** decisions.



### Some detailed explanations

As an example, the developer creates decisions like `channel_choice_1 = germ.Choices([16, 32, 64])`, and uses the decision like `ctx.SearchableConvBNBlock(in_channels=32, out_channels=channel_choice_1 ...)`.

Note that one should either create all searchable blocks in the `with self.begin_searchable()` context, or manually call GermSuperNet.parse_searchable() after module initialization in `__init__`.

The `ctx` is a SearchableContext object `self.ctx`, it is responsible for supplying the architecture rollout in `ctx.rollout` during the search process. The context object `self.ctx` should be passed into every searchable block as the first argument. As a syntactic sugar, one can use `self.ctx.<SearchableBlockClassName>(...)` without `self.ctx` as the first argument.
