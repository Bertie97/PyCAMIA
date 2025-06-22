import os, sys
print(sys.version)
print(os.environ['PATH'])
from pycamia import info_manager

__info__ = info_manager(
    project = "PyCAMIA",
    package = "batorch",
    fileinfo = "The inherited tensor from 'torch' with batch. This is the auto-generator, run the file to automatically generate tensor.py. Please DONOT touch tensor.py, modify tensor_template.py instead. ",
    requires = "torch"
)

from tensorsize import *
from device import AutoDevice
from collections import defaultdict
_device = AutoDevice(verbose=True, always_proceed=True)

with __info__:
    import torch
    from pycamia import scope, alias
    from pycamia import Path, SPrint
    from pycamia import tokenize, token_replace, identity_function
    from pycamia import python_lines, add_lineno
    from pycamia import get_num_indent, tab_to_indent, no_indent, with_indent, get_indent_len
    
with open(Path(__file__).parent / 'tensor_template.py') as fp:
    __text__ = fp.read()
    
    start_sizemap = False
    start_sizemapop = False
    lines_sizemap = []
    lines_sizemapop = []
    for line in python_lines(__text__):
        if line.lstrip().startswith("class SizeMappingFunctions"): start_sizemap = True
        if start_sizemap: lines_sizemap.append(line)
        if line.lstrip().startswith("size_mapping ="): start_sizemap = False
        if line.lstrip().startswith("class OperationSizeMappingFunctions"): start_sizemapop = True
        if start_sizemapop: lines_sizemapop.append(line)
        if line.lstrip().startswith("size_mapping_op ="): start_sizemapop = False
    exec(no_indent('\n'.join(lines_sizemap)))
    exec(no_indent('\n'.join(lines_sizemapop)))
    
    DimKeys = ('function_dim', 'batch_dim', 'sequence_dim', 'feature_dim')

def template(decorators, _def, _fname, _args, doc, cast, get_size, inner_codes, manual_param):
    return f'''
{decorators}
{_def} {_fname}({_args}):
    """
{doc}
    """
    # Cast all arguments to the wanted 'Tensor' or 'Size' type, and convert the dim arguments. 
{cast}
    # Obtain available sized in arguments (which will be fed into size function).  
{get_size}
    # Use the given inner codes if they are provided. 
{inner_codes}
{manual_param}
    if getattr(obj, 'grad_fn', None) is not None:
        obj.grad_fn_name = "{_fname}"
    return obj
    '''

def get_updated_code(codes, mode='function', ignore_args=[]):
    """
    Get codes for inheritance.
    
    Args:
        func (function): the function object to be expanded. 
        mode (str: function|method): the string indicating the block is a function or a method. 
        ignore_arg (list): the list of arguments that are ignored in auto generated codes.
    """
    
    line_no = 0
    func_lines = python_lines(tab_to_indent(codes))
    declaration = func_lines[line_no]
    decorators = []
    while declaration.startswith('@'):
        decorators.append(declaration)
        line_no += 1
        declaration = func_lines[line_no].strip()
    decorators = '\n'.join(decorators)
    
    # Interprete the declaration line of template. 
    _def, _fname, _args, *_tail = tokenize(declaration, sep=[' ', '(', ')', '\n'])
    is_inplace = _fname.endswith('_') and _fname[-2:] != '__'
    line_no += 1
    
    line0 = tokenize(declaration, sep=':', by="()[]{}$$``''\"\"##")[-1].strip()
    
    # Extract the document string from the template. 
    doc_in_template = ''
    if not line0:
        line = func_lines[line_no]
        line_start = line.split('\n', 1)[0].lstrip("\t rf")
        if line_start.startswith('"""') or line_start.startswith("'''") or line_start.startswith('#'):
            doc_in_template = line.strip('\t \'"').strip('\n')
            line_no += 1
    
    # Copy the inner codes from the template. 
    num_indent = get_num_indent(declaration)
    indent = lambda k: " " * get_indent_len() * k
    inner_codes = ""
    if not line0 in ('...', 'pass'):
        inner_code_lines = ([with_indent(line0, 1)] if line0 else []) + func_lines[line_no:]
        res_inner_code_lines = [with_indent("torch_returned = False", 1)]
        if_at_indent = []
        need_else_at_indent = []
        in_else_indent = []
        for inner_line in inner_code_lines:
            nosp_line = inner_line.lstrip(' ')
            if not nosp_line: res_inner_code_lines.append(nosp_line)
            at_indent = get_num_indent(inner_line)
            
            # find 'return'
            has_return = False
            nosp_line_fregs = []
            for x in tokenize(nosp_line, sep=';'):
                if x.lstrip().startswith('return '):
                    has_return = True
                    nosp_line_fregs.append(f'obj = {x[7:]}')
                else: nosp_line_fregs.append(x)
            nosp_line = ';'.join(nosp_line_fregs)
            nosp_line_fregs = []
            for x in tokenize(nosp_line, sep=':'):
                if x.lstrip().startswith('return '):
                    has_return = True
                    nosp_line_fregs.append(f'torch_returned = True; obj = {x.lstrip()[7:]}')
                else: nosp_line_fregs.append(x)
            nosp_line = ': '.join(nosp_line_fregs)
            
            # print(nosp_line, in_else_indent, at_indent, need_else_at_indent)
            while in_else_indent and at_indent < in_else_indent[-1]: in_else_indent.pop(-1)
            while if_at_indent and at_indent < if_at_indent[-1]: if_at_indent.pop(-1)
            if need_else_at_indent and at_indent == need_else_at_indent[-1]:
                if nosp_line.startswith('else'):
                    need_else_at_indent.pop(-1)
                elif nosp_line.startswith('elif'): ...
                else:
                    res_inner_code_lines.append(indent(at_indent + len(in_else_indent)) + 'if not torch_returned:')
                    in_else_indent.append(need_else_at_indent.pop(-1))
            res_inner_code_lines.append(indent(at_indent + len(in_else_indent)) + nosp_line)
            if (nosp_line.startswith('if') or nosp_line.startswith('elif')) and ':' in nosp_line:
                if_at_indent.append(at_indent)
                if has_return:
                    for i in if_at_indent:
                        if i not in need_else_at_indent: need_else_at_indent.append(i)
        inner_codes = '\n'.join(res_inner_code_lines) + '\n'
    if inner_codes.strip() in ('...', 'pass'): inner_codes = ""
    
    # Parse arguments. 
    for ig in ignore_args:
        _args = token_replace(_args, lambda x: x.lstrip('*').startswith(ig), '', sep=[','], strip=' ').replace(',,', ',').rstrip(',')
    
    args_name = None
    kwargs_name = None
    func_args = []
    for part in tokenize(_args, ','):
        default_value = None
        annotation = None
        seq = tokenize(part, '=')
        if len(seq) == 1: var_name = seq[0]
        elif len(seq) == 2: part, default_value = seq
        seq = tokenize(part, ':')
        if len(seq) == 1: var_name = seq[0]
        elif len(seq) == 2: var_name, annotation = seq
        if var_name: var_name = var_name.strip()
        if annotation: annotation = annotation.strip()
        if default_value: default_value = default_value.strip()
        if var_name in "*/": continue
        if var_name.startswith('**'): kwargs_name = var_name.lstrip('*')
        elif var_name.startswith('*'): args_name = var_name.lstrip('*')
        func_args.append((var_name.lstrip('*'), annotation, default_value))
    
    # Classify the arguments. 
    self_name = func_args[0][0]
    
    tensor_args = []
    size_args = []
    dim_args = []
    iter_dim_args = []
    rev_dim_args = []
    rmv_star_args = []
    
    inherit_args = []
    inherit_kwargs = []
    
    for name, annot, default in func_args:
        if annot is None: annot = ''
        if annot.endswith('[:]'): dim_args.append(name); rmv_star_args.append(name)
        elif annot.endswith(']') and 'dim[' in annot:
            dim_args.append(name)
            iter_dim_args.append(name)
            if annot.startswith('del_dim['):
                rev_dim_args.append(name)
        elif annot.strip('\'"') == 'Tensor': tensor_args.append(name)
        elif annot.strip('\'"') == 'Size': size_args.append(name)
        elif annot in "new_dim del_dim exist_dim".split(): dim_args.append(name)
        elif annot.startswith('linalg_dim'): dim_args.append(name)
        
        if name == args_name: inherit_kwargs.append('*' + name)
        elif name == kwargs_name: inherit_kwargs.append('**' + name)
        elif default is not None: inherit_kwargs.append(f"{name}={name}")
        elif mode == 'function' or name != self_name: inherit_args.append(name)
    inherit_args = ','.join(inherit_args)
    inherit_kwargs = ','.join(inherit_kwargs)
    
    # Generate casting codes. 
    annot_dict = {n: a for n, a, d in func_args}
    tensor_cast = lambda x: no_indent(f"""
        if {x} is None: ...
        elif not isinstance({x}, torch.Tensor): {x} = torch.tensor({x})
        if not isinstance({x}, subclass): {x} = {x}.as_subclass(subclass).special_from({x}.shape)
        if pivot is not None and hasattr(pivot, 'device') and pivot.device.type != 'cpu':
            if {x}.device.type == 'cpu': {x} = {x}.to(pivot.device)
    """)
    dim_cast = lambda d: f"{d} = {annot_dict[d]}({self_name}, {'*' if d == args_name else ''}{d})"
    
    all_tensor_cast = '\n'.join([tensor_cast(x) for x in tensor_args])
    dim_tensor_cast = '\n'.join([dim_cast(d) for d in dim_args])
    cast = with_indent(no_indent(f"""
pivot = None
for t in [{', '.join(tensor_args)}]:
    if isinstance(t, torch.Tensor): pivot = t; break
subclass = Tensor.get_tensor_subclass(pivot)

{all_tensor_cast}
{dim_tensor_cast}
    """), 1)
    
    get_size = with_indent('\n'.join([f"{x}_shape=None if {x} is None else Size({x}.shape)" for x in tensor_args] + [f'{x}=Size(%s{x})'%('*' if x == args_name else '') for x in size_args]), 1)
    size_fname = _fname
    if is_inplace: size_fname = size_fname.rstrip('_')
    size_arguments = ', '.join([f'{x}_shape' for x in tensor_args] + size_args + dim_args)
    
    # Auto generate auto 'inner_codes' for cases where no inner-codes are provided. 
    kwargs_dict = f"dict({', '.join(f'{n}={n}' for n, a, d in func_args if d is not None)})"
    size_map_func = 'size_mapping_op' if len(tensor_args) >= 2 else 'size_mapping'
    kwarg = f", **{kwargs_dict}" if eval(size_map_func)[size_fname].__code__.co_flags & 0x08 else ''
    ref_shape_defined = False
    if not inner_codes:
        if len(tensor_args) >= 2:
            inner_codes = indent(1) + "ref_shape, " + ', '.join([f'{x}_shape' for x in tensor_args]) + \
                f" = {size_map_func}['{size_fname}']({size_arguments}{kwarg})\n"
            for x in tensor_args: inner_codes += indent(1) + f"{x} = {x}.view({x}_shape)\n"
            ref_shape_defined = True
        
        parent = f"torch_super({self_name}, '{_fname}')" if mode == 'method' else f'torch.{_fname}'
        if len(iter_dim_args) > 0:
            lists = ', '.join([iarg + '[::-1]' if iarg in rev_dim_args else iarg for iarg in iter_dim_args])
            iters = ', '.join([d + '_iter' for d in iter_dim_args])
            if len(iter_dim_args) == 1: iters += ','
            for d in iter_dim_args:
                inherit_args = token_replace(inherit_args, d, d + "_iter", sep=[',', '='], strip=' ')
                inherit_kwargs = token_replace(inherit_kwargs, f"{d}={d}", f"{d}={d}_iter", sep=[','], strip=' ')
                if d == args_name:
                    inherit_kwargs = token_replace(inherit_kwargs, lambda x: x == '*' + args_name, d+'_iter', sep=[',', '='], strip=' ')
            inner_codes += indent(1) + "with torch._C.DisableTorchFunction():\n"
            inner_codes += indent(2) + f"for {iters} in zip({lists}):\n"
            inner_codes += indent(3) + f"auto_generated_args = tuple(x for x in [{inherit_args}] if x is not None)\n"
            if is_inplace: inner_codes += indent(3) + f"{parent}(*auto_generated_args, {inherit_kwargs})\n"
            else: inner_codes += indent(3) + f"{self_name} = {parent}(*auto_generated_args, {inherit_kwargs})\n"
            inner_codes += indent(1) + f"obj = {self_name}\n"
        else:
            for d in rmv_star_args:
                inherit_kwargs = token_replace(inherit_kwargs, lambda x: x.lstrip('*') == d, d, sep=[',', '='], strip=' ')
            inner_codes += indent(1) + "with torch._C.DisableTorchFunction():\n"
            inner_codes += indent(2) + f"auto_generated_args = tuple(x for x in [{inherit_args}] if x is not None)\n"
            if is_inplace:
                inner_codes += indent(2) + f"{parent}(*auto_generated_args, {inherit_kwargs})\n"
                inner_codes += indent(2) + f"obj = {self_name}\n"
            else: inner_codes += indent(2) + f"obj = {parent}(*auto_generated_args, {inherit_kwargs})\n"   
    if 'special_from' not in inner_codes and 'Tensor.inherit_from' not in inner_codes:
        if not ref_shape_defined:
            inner_codes += indent(1) + f"ref_shape{', *_' if size_map_func == 'size_mapping_op' else ''} = {size_map_func}['{size_fname}']({size_arguments}{kwarg})\n"
        if is_inplace: inner_codes += indent(1) + f"obj.special_from(ref_shape)"
        else: inner_codes += indent(1) + f"obj = Tensor.inherit_from(obj, pivot, shape=...).special_from(ref_shape, allow_view=True)"
    
    # Deal with **kwargs:
    if kwargs_name is None:
        _args += ''.join(f", {k}=None" for k in DimKeys)
        props = ', '.join(f'{k}={k}' for k in DimKeys)
        manual_param = with_indent(f"""
        def update_special(obj, updates):
            if isinstance(obj, tuple):
                for x in obj: update_special(x, updates)
            elif isinstance(obj, Tensor): obj.special_dims.update(updates)
        update_special(obj, {{k: v for k, v in dict({props}).items() if v != None}})
        """, 1)
    else:
        props = ', '.join(f"{k}={kwargs_name}.get('{k}', None)" for k in DimKeys)
        manual_param = with_indent(f"""
        def update_special(obj, updates):
            if isinstance(obj, tuple):
                for x in obj: update_special(x, updates)
            elif isinstance(obj, Tensor): obj.special_dims.update(updates)
        update_special(obj, {{k: v for k, v in dict({props}).items() if v != None}})
        """, 1)
    
    func_codes = template(decorators, _def, _fname, _args, doc_in_template if doc_in_template else with_indent('***', 2), cast, get_size, inner_codes, manual_param)
    resulting_lines = []
    for line in [l.rstrip() for l in func_codes.split('\n')]:
        if line == '' and (len(resulting_lines) == 0 or resulting_lines[-1] == ''): continue
        resulting_lines.append(line)
    func_codes = '\n'.join(resulting_lines).replace('"""', "'''")
    
    # Generate new document string. 
    torch_doc = ''
    if mode=='method': parent = torch.Tensor
    else: parent = torch
    if hasattr(parent, _fname):
        torch_doc = f"The document of the original function is:\n{getattr(parent, _fname).__doc__}"
        for line in torch_doc.split('\n'):
            if line.startswith("See :func:"):
                torch_doc += f"\nwhich is:\n{eval(line.split('See :func:')[-1].strip().strip('`.')).__doc__}"
                break
    new_doc = with_indent(f"""{no_indent(doc_in_template)}

Automatically inheritted method from '{'torch.Tensor' if mode=='method' else 'torch'}.{_fname}'. The automatically generated codes are as follows:
{with_indent(add_lineno(no_indent(func_codes)), 1)}
{with_indent(torch_doc, 1)}
        """, 1)
        
    func_codes = template(decorators, _def, _fname, _args, new_doc, cast, get_size, inner_codes, manual_param)
    resulting_lines = []
    for line in [l.rstrip() for l in func_codes.split('\n')]:
        if line == '' and (len(resulting_lines) == 0 or resulting_lines[-1] == ''): continue
        resulting_lines.append(line)
    return '\n'.join(resulting_lines)

with open(Path(__file__).parent / 'tensor_template.py') as fp:
    __lines__ = fp.read().split('\n')
    sprint = SPrint()
    
def func_section(func_lines, state, ignore_out=False, n_indent=0):
    new_codes = with_indent(get_updated_code(
        no_indent('\n'.join(func_lines)), 
        mode='function' if state == 'global' else 'method', 
        ignore_args=['out'] if ignore_out else []
    ), n_indent)
    # if new_codes.strip():
        # print(no_indent('\n'.join(func_lines)), end='\n' + '~' * 80 + '\n')
        # print(new_codes, end='\n' + '~' * 80 + '\n')
    return new_codes

with scope("main"):
    state = "normal" # In: normal, method, global, biway-method, biway-global
    n_indent = 0
    biway_section = ""
    func_lines = []
    in_func = False
    for line in __lines__:
        c_indent = get_num_indent(line)
        
        if state == 'biway-method':
            if len(func_lines) > 0:
                sprint(func_section(func_lines, 'method', n_indent=n_indent))
            sprint(with_indent(biway_section, n_indent))
            state = 'normal'
            continue
        
        start_dec = c_indent == n_indent and line.lstrip().startswith('@')
        start_func = c_indent == n_indent and line.lstrip().startswith('def ')
        if in_func:
            end_func = state == 'normal' or c_indent <= n_indent and line.strip()
            if not end_func:
                if start_func: in_func = True
                func_lines.append(line); continue
            # Do when end function
            sprint(func_section(func_lines, state.split('-')[-1], n_indent=n_indent))
            if state == 'biway-global':
                biway_section += func_section(func_lines, 'method', ignore_out=True, n_indent=n_indent) + '\n'
            in_func = False
            func_lines = []
        if state == 'normal': sprint(line)
        elif start_dec: func_lines = [line]
        elif start_func: in_func = True; func_lines.append(line)
        else: sprint(line)
        
        if line.strip() == "### START BIWAY AUTO GENERATION":    n_indent = c_indent; state = 'biway-global'; continue
        elif line.strip() == "### START METHOD AUTO GENERATION": n_indent = c_indent; state = 'method';       continue
        elif line.strip() == "### START GLOBAL AUTO GENERATION": n_indent = c_indent; state = 'global';       continue
        elif line.strip() == "### STOP BIWAY AUTO GENERATION":   n_indent = c_indent; state = 'normal';       continue
        elif line.strip() == "### STOP METHOD AUTO GENERATION":  n_indent = c_indent; state = 'biway-method'; continue
        elif line.strip() == "### STOP GLOBAL AUTO GENERATION":  n_indent = c_indent; state = 'normal';       continue

with open(Path(__file__).parent / 'tensor.py', 'w') as fp:
    fp.write(sprint.text)
