import re
import uuid
from typing import List, Dict, Union, Optional


Attributes = Dict[str, str]
Element = Dict[str, Union[str, Attributes, List['Element']]]
LPMLTree = List[Union[str, Element]]


PATTERN_ATTRIBUTE = r''' ([^"'/<> -]+)=(?:"([^"]*)"|'([^']*)')'''
PATTERN_ATTRIBUTE_NO_CAPTURE = r''' [^"'/<> -]+=(?:"[^"]*"|'[^']*')'''
PATTERN_TAG_START = rf'<([^/>\s\n]+)((?:{PATTERN_ATTRIBUTE_NO_CAPTURE})*)\s*>'
PATTERN_TAG_END = r'</([^/>\s\n]+)\s*>'
PATTERN_TAG_EMPTY = rf'<([^/>\s\n]+)((?:{PATTERN_ATTRIBUTE_NO_CAPTURE})*)\s*/>'
PATTERN_TAG = rf'({PATTERN_TAG_START}|{PATTERN_TAG_END}|{PATTERN_TAG_EMPTY})'
PATTERN_BACKTICK = r'`(.*?)`'


def _parse_attributes(text: str) -> Attributes:
    attributes: Attributes = {}
    for k, v1, v2 in re.findall(PATTERN_ATTRIBUTE, text):
        attributes[k] = v1 or v2
    return attributes


def _restore_protected_content(
        tree: LPMLTree, protected: Dict[str, str]) -> LPMLTree:
    """Recursively traverse the tree and restore placeholders."""
    restored_tree: LPMLTree = []
    for item in tree:
        if isinstance(item, str):
            # Restore all placeholders found in the string content
            for placeholder, original in protected.items():
                item = item.replace(placeholder, original)
            restored_tree.append(item)
        elif isinstance(item, dict):
            if isinstance(item['content'], list):
                item['content'] = _restore_protected_content(
                    item['content'], protected)
            restored_tree.append(item)
    return restored_tree


def parse(text: str, strip: bool = False, 
          exclude: Optional[List[str]] = None) -> LPMLTree:
    """Parse LPML text.

    Args:
        text (str): The text to parse.
        exclude (List[str]): Content of the specified tags will not be parsed.

    Returns:
        LPMLTree: The parsed tree.
    """
    protected_content: Dict[str, str] = {}

    # 1. Protect phase: Replace backticked content with unique placeholders
    def protect_match(match):
        # Generate a unique placeholder that is extremely unlikely to collide
        placeholder = f"__PROTECTED_{uuid.uuid4().hex}__"
        # Store the original content (including backticks)
        protected_content[placeholder] = match.group(0)
        return placeholder

    text = re.sub(
        PATTERN_BACKTICK, protect_match, text, flags=re.DOTALL)

    if exclude is None:
        exclude = []

    tree: LPMLTree = []

    cursor = 0
    tag_exclude = None
    stack = [{'tag': 'root', 'content': tree}]

    for match in re.finditer(PATTERN_TAG, text):
        tag = match.group(0)
        match_tag_start = re.fullmatch(PATTERN_TAG_START, tag)
        match_tag_end = re.fullmatch(PATTERN_TAG_END, tag)
        match_tag_empty = re.fullmatch(PATTERN_TAG_EMPTY, tag)

        if tag_exclude is not None:
            if match_tag_end and match_tag_end.group(1) == tag_exclude:
                tag_exclude = None
            else:
                continue

        ind_tag_start, ind_tag_end = match.span()
        content_str = text[cursor:ind_tag_start]
        if strip:
            content_str = content_str.strip()
        if content_str:
            stack[-1]['content'].append(content_str)
        cursor = ind_tag_end

        if match_tag_start:
            name = match_tag_start.group(1)
            if name in exclude:
                tag_exclude = name

            attributes = _parse_attributes(match_tag_start.group(2))
            element: Element = {
                'tag': name,
                'attributes': attributes,
                'content': []
            }
            stack[-1]['content'].append(element)
            stack.append(element)

        elif match_tag_empty:
            name = match_tag_empty.group(1)
            attributes = _parse_attributes(match_tag_empty.group(2))
            element: Element = {
                'tag': name,
                'attributes': attributes,
                'content': None
            }
            stack[-1]['content'].append(element)

        elif match_tag_end:
            name = match_tag_end.group(1)
            for i in range(len(stack)-1, 0, -1):
                if stack[i]['tag'] == name:
                    ind_tag_start = i
                    break
            else:
                print(f'Warning: Unmatched closing tag </{name}> found.')
                stack[-1]['content'].append(tag)
            stack = stack[:max(1, ind_tag_start)]

    content_str = text[cursor:]
    if strip:
        content_str = content_str.strip()
    if content_str:
        stack[-1]['content'].append(content_str)

    if len(stack) > 1:
        tags_remain = [e["tag"] for e in stack][1:]
        print(f'Warning: Unclosed elements remain: {tags_remain}')

    tree = _restore_protected_content(tree, protected_content)
    return tree


def _repr_tag(tag, content, **kwargs):
    if kwargs:
        attr = ' ' + ' '.join([f'{k}="{v}"' for k, v in kwargs.items()])
    else:
        attr = ''
    bra = f'<{tag}{attr}>'
    ket = f'</{tag}>'
    emp = f'<{tag}/>'
    if content is None:
        return emp
    return ''.join([bra, content, ket])


def deparse(tree: LPMLTree) -> str:
    """Deparse LPML tree.

    Args:
        tree (LPMLTree): The tree to deparse.

    Returns:
        str: The deparsed text.
    """
    if tree is None:
        return tree

    text = ''

    for element in tree:
        if isinstance(element, str):
            text += element
            continue
        deparsed_content = deparse(element['content'])
        text += _repr_tag(
            element['tag'], deparsed_content, **element['attributes'])
    return text


def findall(tree: LPMLTree, tag: str) -> List[Element]:
    """Find all elements with the specified tag.

    Args:
        tree (LPMLTree): The tree to search.
        tag (str): The tag to search for.

    Returns:
        List[Element]: The list of elements with the specified tag.
    """
    if tree is None:
        return []

    result = []
    for element in tree:
        if not isinstance(element, dict):
            continue
        if element['tag'] == tag:
            result.append(element)
        result += findall(element['content'], tag)
    return result


def generate_element(tag: str, content: str, **attributes) -> Element:
    return {
        'tag': tag,
        'attributes': attributes,
        'content': content
    }
