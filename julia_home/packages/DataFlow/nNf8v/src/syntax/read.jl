# Syntax → Graph

function bindings(ex)
  bs = []
  for ex in ex.args
    @capture(ex, x_ = _) && push!(bs, x)
  end
  return bs
end

function normreturn(ex)
  @capture(ex.args[end], return x_) && (ex.args[end] = x)
  ex
end

function normedges(ex)
  ex = normreturn(ex)
  map!(ex.args, ex.args) do ex
    isline(ex) ? ex :
    @capture(ex, _ = _) ? ex :
    :($(gensym("edge")) = $ex)
  end
  return ex
end

normblock(ex) = @> ex MacroTools.flatten block normedges normlines

function latenodes(exs)
  bindings = d()
  for ex in exs
    @capture(ex, b_Symbol = x_)
    bindings[b] = dvertex(x)
  end
  return bindings
end

graphm(bindings, node) = constant(node)
graphm(bindings, node::Vertex) = node
graphm(bindings, ex::Symbol) =
  haskey(bindings, ex) ? graphm(bindings, bindings[ex]) : constant(ex)

function tovertex!(v, bs, f, xs...)
  v.value = f
  thread!(v, map(ex -> graphm(bs, ex), xs)...)
end

function graphm!(v, bindings, ex::Expr)
  if isexpr(ex, :vertex)
    tovertex!(v, bindings, ex.args...)
  elseif @capture(ex, f_(args__))
    tovertex!(v, bindings, f, args...)
  else
    v.value = Constant(ex)
  end
  return v
end

graphm(bindings, ex::Expr) =
  isexpr(ex, :block) ? graphm(bindings, ex.args) :
    graphm!(dvertex(nothing), bindings, ex)

function fillnodes!(bindings, nodes)
  nodes′ = []
  for b in nodes
    node = bindings[b]
    if !(value(node) isa Expr)
      bindings[b] = graphm(bindings, value(node))
    else
      push!(nodes′, b)
    end
  end
  for b in nodes′
    node = bindings[b]
    graphm!(node, bindings, node.value)
  end
  return bindings
end

function graphm(bindings, exs::Vector)
  exs = normblock(:($(exs...);)).args
  @capture(exs[end], result_Symbol = _)
  lates = latenodes(exs)
  merge!(bindings, lates)
  fillnodes!(bindings, keys(lates))
  output = graphm(bindings, result)
end

bindargs(xs) = Dict{Any,Any}(x => inputnode(i) for (i, x) in enumerate(xs))

normalise(ex) =
  @> ex normops normcalls normsplits normclosures desugar

graphm(x; args = ()) = graphm(bindargs(args), normalise(x))
