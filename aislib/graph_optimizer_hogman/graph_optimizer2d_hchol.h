// HOG-Man - Hierarchical Optimization for Pose Graphs on Manifolds
// Copyright (C) 2010 G. Grisetti, R. Kümmerle, C. Stachniss
// 
// HOG-Man is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// HOG-Man is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef _GRAPH_OPTIMIZER2D_HCHOL_HH_
#define _GRAPH_OPTIMIZER2D_HCHOL_HH_

#include "graph_optimizer_hchol.h"
#include "graph_optimizer/graph_optimizer2d.h"
#include "math/transformation.h"

namespace AISNavigation{

  /**
   * \brief 2D hirachical cholesky optimizer
   */
  typedef HCholOptimizer<PoseGraph2D> HCholOptimizer2D;

}
#endif
