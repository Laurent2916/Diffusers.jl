using Flux

struct SinusoidalPositionEmbedding{W<:AbstractArray}
  weight::W
end

Flux.@functor SinusoidalPositionEmbedding
Flux.trainable(emb::SinusoidalPositionEmbedding) = () # mark it as an non-trainable array

function SinusoidalPositionEmbedding(in::Integer, out::Integer)
  W = make_positional_embedding(out, in)
  SinusoidalPositionEmbedding(W)
end

function make_positional_embedding(dim_embedding::Integer, seq_length::Integer=1000; n::Integer=10000)
  embedding = Matrix{Float32}(undef, dim_embedding, seq_length)
  for pos in 1:seq_length
    for row in 0:2:(dim_embedding-1)
      denom = 1.0 / (n^(row / (dim_embedding - 2)))
      embedding[row+1, pos] = sin(pos * denom)
      embedding[row+2, pos] = cos(pos * denom)
    end
  end
  embedding
end

(m::SinusoidalPositionEmbedding)(x::Integer) = m.weight[:, x]
(m::SinusoidalPositionEmbedding)(x::AbstractVector) = NNlib.gather(m.weight, x)
(m::SinusoidalPositionEmbedding)(x::AbstractArray) = reshape(m(vec(x)), :, size(x)...)

function Base.show(io::IO, m::SinusoidalPositionEmbedding)
  print(io, "SinusoidalPositionEmbedding(", size(m.weight, 2), " => ", size(m.weight, 1), ")")
end
