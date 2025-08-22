export type Bounds = { x: number; y: number; width: number; height: number }
export type Point = { x: number; y: number }

export class QuadTree {
  private bounds: Bounds
  private maxObjects: number
  private maxLevels: number
  private level: number
  private objects: Point[] = []
  private nodes: QuadTree[] = []

  constructor(bounds: Bounds, maxObjects = 10, maxLevels = 4, level = 0) {
    this.bounds = bounds
    this.maxObjects = maxObjects
    this.maxLevels = maxLevels
    this.level = level
  }

  clear(): void {
    this.objects = []
    this.nodes = []
  }

  split(): void {
    const subWidth = this.bounds.width / 2
    const subHeight = this.bounds.height / 2
    const x = this.bounds.x
    const y = this.bounds.y

    this.nodes[0] = new QuadTree(
      { x: x + subWidth, y: y, width: subWidth, height: subHeight },
      this.maxObjects,
      this.maxLevels,
      this.level + 1
    )

    this.nodes[1] = new QuadTree(
      { x: x, y: y, width: subWidth, height: subHeight },
      this.maxObjects,
      this.maxLevels,
      this.level + 1
    )

    this.nodes[2] = new QuadTree(
      { x: x, y: y + subHeight, width: subWidth, height: subHeight },
      this.maxObjects,
      this.maxLevels,
      this.level + 1
    )

    this.nodes[3] = new QuadTree(
      { x: x + subWidth, y: y + subHeight, width: subWidth, height: subHeight },
      this.maxObjects,
      this.maxLevels,
      this.level + 1
    )
  }

  getIndex(point: Point): number {
    let index = -1
    const verticalMidpoint = this.bounds.x + this.bounds.width / 2
    const horizontalMidpoint = this.bounds.y + this.bounds.height / 2

    const topQuadrant = point.y < horizontalMidpoint && point.y + 0 > horizontalMidpoint
    const bottomQuadrant = point.y > horizontalMidpoint

    if (point.x < verticalMidpoint && point.x + 0 < verticalMidpoint) {
      if (topQuadrant) {
        index = 1
      } else if (bottomQuadrant) {
        index = 2
      }
    } else if (point.x > verticalMidpoint) {
      if (topQuadrant) {
        index = 0
      } else if (bottomQuadrant) {
        index = 3
      }
    }

    return index
  }

  insert(point: Point): void {
    if (this.nodes.length > 0) {
      const index = this.getIndex(point)
      if (index !== -1) {
        this.nodes[index].insert(point)
        return
      }
    }

    this.objects.push(point)

    if (this.objects.length > this.maxObjects && this.level < this.maxLevels) {
      if (this.nodes.length === 0) {
        this.split()
      }

      let i = 0
      while (i < this.objects.length) {
        const index = this.getIndex(this.objects[i])
        if (index !== -1) {
          this.nodes[index].insert(this.objects.splice(i, 1)[0])
        } else {
          i++
        }
      }
    }
  }

  retrieve(point: Point, radius: number): Point[] {
    const found: Point[] = []
    const searchArea = {
      x: point.x - radius,
      y: point.y - radius,
      width: radius * 2,
      height: radius * 2
    }

    this._retrieve(searchArea, found)
    return found
  }

  private _retrieve(searchArea: Bounds, found: Point[]): void {
    if (!this._intersects(searchArea, this.bounds)) {
      return
    }

    for (const obj of this.objects) {
      if (this._intersects(searchArea, { x: obj.x, y: obj.y, width: 0, height: 0 })) {
        found.push(obj)
      }
    }

    if (this.nodes.length > 0) {
      for (const node of this.nodes) {
        node._retrieve(searchArea, found)
      }
    }
  }

  private _intersects(rect1: Bounds, rect2: Bounds): boolean {
    return !(
      rect2.x > rect1.x + rect1.width ||
      rect2.x + rect2.width < rect1.x ||
      rect2.y > rect1.y + rect1.height ||
      rect2.y + rect2.height < rect1.y
    )
  }

  getBounds(): Bounds {
    return { ...this.bounds }
  }

  getObjectCount(): number {
    let count = this.objects.length
    for (const node of this.nodes) {
      count += node.getObjectCount()
    }
    return count
  }
}
