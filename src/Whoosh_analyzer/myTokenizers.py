from whoosh.compat import u, text_type
from whoosh.analysis.acore import Composable, Token
from whoosh.util.text import rcompile
import boostwords

default_pattern = rcompile(r"\w+(\.?\w+)*")
booster = boostwords.BOOST_VALUES
    
class Tokenizer(Composable):
    """Base class for Tokenizers.
    """

    def __eq__(self, other):
        return other and self.__class__ is other.__class__

class myRegexTokenizer(Tokenizer):
    """
    Uses a regular expression to extract tokens from text, 
    assigning custom boost values to token
    """

    def __init__(self, expression=default_pattern, booster=booster,gaps=False):
        """
        :param expression: A regular expression object or string. Each match
            of the expression equals a token. Group 0 (the entire matched text)
            is used as the text of the token. If you require more complicated
            handling of the expression match, simply write your own tokenizer.
        :param gaps: If True, the tokenizer *splits* on the expression, rather
            than matching on the expression.
        """

        self.expression = rcompile(expression)
        self.gaps = gaps
        self.booster = booster

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            if self.expression.pattern == other.expression.pattern:
                return True
        return False

    def __call__(self, value, positions=False, chars=False, keeporiginal=False,
                 removestops=True, start_pos=0, start_char=0, tokenize=True, booster=booster,
                 mode='', **kwargs):
        """
        :param value: The unicode string to tokenize.
        :param positions: Whether to record token positions in the token.
        :param chars: Whether to record character offsets in the token.
        :param start_pos: The position number of the first token. For example,
            if you set start_pos=2, the tokens will be numbered 2,3,4,...
            instead of 0,1,2,...
        :param start_char: The offset of the first character of the first
            token. For example, if you set start_char=2, the text "aaa bbb"
            will have chars (2,5),(6,9) instead (0,3),(4,7).
        :param tokenize: if True, the text should be tokenized.
        """

        assert isinstance(value, text_type), "%s is not unicode" % repr(value)

        t = Token(positions, chars, removestops=removestops, mode=mode, 
                  **kwargs)
        
        if not tokenize:
            t.original = t.text = value
            t.boost = 1.0
            if t.text in booster:
                t.boost = booster[t.text]           
            if positions:
                t.pos = start_pos
            if chars:
                t.startchar = start_char
                t.endchar = start_char + len(value)
            yield t
        elif not self.gaps:
            # The default: expression matches are used as tokens
            for pos, match in enumerate(self.expression.finditer(value)):
                t.text = match.group(0)
                t.boost = 1.0
                if t.text in booster:
                    t.boost = booster[t.text]                   
                if keeporiginal:
                    t.original = t.text
                t.stopped = False
                if positions:
                    t.pos = start_pos + pos
                if chars:
                    t.startchar = start_char + match.start()
                    t.endchar = start_char + match.end()
                yield t
        else:
            # When gaps=True, iterate through the matches and
            # yield the text between them.
            prevend = 0
            pos = start_pos
            for match in self.expression.finditer(value):
                start = prevend
                end = match.start()
                text = value[start:end]
                if text:
                    t.text = text
                    t.boost = 1.0
                    if t.text in booster:
                        t.boost = booster[t.text] 
                    if keeporiginal:
                        t.original = t.text
                    t.stopped = False
                    if positions:
                        t.pos = pos
                        pos += 1
                    if chars:
                        t.startchar = start_char + start
                        t.endchar = start_char + end

                    yield t

                prevend = match.end()

            # If the last "gap" was before the end of the text,
            # yield the last bit of text as a final token.
            if prevend < len(value):
                t.text = value[prevend:]
                t.boost = 1.0
                if t.text in booster:
                    t.boost = booster[t.text] 
                if keeporiginal:
                    t.original = t.text
                t.stopped = False
                if positions:
                    t.pos = pos
                if chars:
                    t.startchar = prevend
                    t.endchar = len(value)
                yield t
