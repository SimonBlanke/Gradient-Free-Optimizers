"""Sphinx extension: add parameter entries to the page TOC.

Numpydoc renders parameters as definition lists inside a field-list
entry ("Parameters: ...").  These are invisible to Sphinx's built-in
TOC collector, which only recognises section nodes and
``desc_signature`` nodes.

This extension runs *after* ``TocTreeCollector`` (priority 900 > 500),
adds anchor ids to every parameter ``<dt>`` in the doctree, and injects
matching entries into ``env.tocs[docname]`` so the page-toc sidebar
picks them up.

Handles both class constructors (grouped under a "Parameters" heading)
and methods (parameters nested directly under the method entry).
"""

from __future__ import annotations

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx

# -- helpers -----------------------------------------------------------------


def _get_obj_id(desc_node: nodes.Element) -> str | None:
    """Return the primary anchor id from a ``desc_signature``."""
    for sig in desc_node.traverse(addnodes.desc_signature):
        ids = sig.get("ids", [])
        if ids:
            return ids[0]
    return None


def _collect_parameters(
    desc_node: nodes.Element,
    obj_id: str,
) -> tuple[str, list[tuple[str, str]]] | None:
    """Walk a desc node and extract numpydoc parameter entries.

    Returns ``(parameters_field_anchor, [(param_name, param_anchor), ...])``
    or ``None`` when there is no Parameters field.
    """
    for field_list in desc_node.traverse(nodes.field_list):
        for field in field_list.children:
            if not isinstance(field, nodes.field):
                continue
            if field.children[0].astext().strip() != "Parameters":
                continue

            section_id = f"{obj_id}-parameters"
            field["ids"].append(section_id)

            params: list[tuple[str, str]] = []
            field_body = field.children[1]
            for dl in field_body.traverse(nodes.definition_list):
                for dli in dl.children:
                    if not isinstance(dli, nodes.definition_list_item):
                        continue
                    term = dli.children[0]
                    for child in term.children:
                        if isinstance(child, nodes.strong):
                            name = child.astext()
                            pid = f"{obj_id}-p-{name}"
                            term["ids"].append(pid)
                            params.append((name, pid))
                            break

            if params:
                return section_id, params

    return None


def _make_toc_item(
    text: str,
    docname: str,
    anchor_id: str,
    *,
    code: bool = False,
) -> nodes.list_item:
    """Build a single TOC ``list_item`` with a reference.

    Parameters
    ----------
    text : str
        Display label shown in the sidebar.
    docname : str
        Document name the reference points to.
    anchor_id : str
        Target anchor (without leading ``#``).
    code : bool
        Wrap *text* in a ``literal`` node (monospace) like method/attribute
        entries do.
    """
    ref = nodes.reference(
        "",
        "",
        internal=True,
        refuri=docname,
        anchorname=f"#{anchor_id}",
    )
    if code:
        ref.append(nodes.literal("", text))
    else:
        ref.append(nodes.Text(text))
    para = addnodes.compact_paragraph("", "", ref)
    return nodes.list_item("", para)


# -- event handler -----------------------------------------------------------


def _process_doctree(app: Sphinx, doctree: nodes.document) -> None:
    """Inject parameter anchors into the doctree and TOC entries."""
    env = app.env
    docname = env.docname

    if docname not in env.tocs:
        return

    # Phase 1 -- collect parameters and add anchor ids to the doctree
    # key: anchor id, value: (objtype, section_id, params)
    collected: dict[str, tuple[str, str, list[tuple[str, str]]]] = {}

    for desc_node in doctree.traverse(addnodes.desc):
        objtype = desc_node.get("objtype")
        if objtype not in ("class", "method"):
            continue
        obj_id = _get_obj_id(desc_node)
        if not obj_id:
            continue
        result = _collect_parameters(desc_node, obj_id)
        if result:
            section_id, params = result
            collected[obj_id] = (objtype, section_id, params)

    if not collected:
        return

    # Phase 2 -- patch env.tocs so the page-toc sidebar shows them
    toc = env.tocs[docname]

    for list_item in toc.traverse(nodes.list_item):
        if not list_item.children:
            continue
        first = list_item.children[0]
        if not isinstance(first, addnodes.compact_paragraph):
            continue
        for ref in first.children:
            if not isinstance(ref, nodes.reference):
                continue
            anchor = ref.get("anchorname", "").lstrip("#")
            if anchor not in collected:
                continue

            objtype, section_id, params = collected[anchor]

            if objtype == "class":
                # locate (or create) the nested bullet list
                nested = None
                for child in list_item.children:
                    if isinstance(child, nodes.bullet_list):
                        nested = child
                        break
                if nested is None:
                    nested = nodes.bullet_list()
                    list_item.append(nested)

                # "Parameters" group with individual params nested below
                group_item = _make_toc_item(
                    "Parameters",
                    docname,
                    section_id,
                    code=False,
                )
                param_list = nodes.bullet_list()
                for pname, pid in params:
                    param_list.append(_make_toc_item(pname, docname, pid, code=True))
                group_item.append(param_list)
                nested.insert(0, group_item)

            elif objtype == "method":
                # nest params directly under the method entry
                param_list = nodes.bullet_list()
                for pname, pid in params:
                    param_list.append(_make_toc_item(pname, docname, pid, code=True))
                list_item.append(param_list)

            break


# -- setup -------------------------------------------------------------------


def setup(app: Sphinx) -> dict:
    # priority 900 ensures we run *after* TocTreeCollector (priority 500)
    app.connect("doctree-read", _process_doctree, priority=900)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
