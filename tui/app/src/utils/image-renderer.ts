/**
 * Image rendering utility for terminal display.
 * Uses half-block characters (▀▄) to render images in the terminal.
 * Each terminal cell represents 2 vertical pixels (top and bottom).
 */

// @ts-ignore - jimp types may not be installed yet
import { Jimp } from "jimp"
import type { OptimizedBuffer } from "@opentui/core"
import { RGBA } from "@opentui/core"

export type ImageData = {
  pixels: Uint8Array // RGBA format, 4 bytes per pixel
  width: number // in pixels
  height: number // in pixels
}

// Half-block characters for terminal rendering
const UPPER_HALF_BLOCK = "▀" // U+2580 - top pixel is foreground
const SPACE = " " // both pixels are background color

/**
 * Load an image from a URL (presigned S3 URL or data URL).
 */
export async function loadImageFromUrl(url: string): Promise<ImageData | null> {
  try {
    if (url.startsWith("data:")) {
      return loadImageFromBase64(url)
    }

    const response = await fetch(url)
    if (!response.ok) {
      console.error(`Failed to fetch image: HTTP ${response.status}`)
      return null
    }

    const buffer = await response.arrayBuffer()
    const image = await Jimp.read(Buffer.from(buffer))

    return {
      pixels: new Uint8Array(image.bitmap.data),
      width: image.bitmap.width,
      height: image.bitmap.height,
    }
  } catch (error) {
    console.error("Error loading image from URL:", error)
    return null
  }
}

/**
 * Load an image from a base64 data URL.
 */
export async function loadImageFromBase64(dataUrl: string): Promise<ImageData | null> {
  try {
    // Extract base64 data from data URL
    const match = dataUrl.match(/^data:image\/[^;]+;base64,(.+)$/)
    if (!match) {
      console.error("Invalid data URL format")
      return null
    }

    const base64Data = match[1]
    const buffer = Buffer.from(base64Data, "base64")
    const image = await Jimp.read(buffer)

    return {
      pixels: new Uint8Array(image.bitmap.data),
      width: image.bitmap.width,
      height: image.bitmap.height,
    }
  } catch (error) {
    console.error("Error loading image from base64:", error)
    return null
  }
}

/**
 * Scale an image to fit within terminal cell dimensions.
 * Accounts for half-block rendering (2 pixels per cell vertically).
 *
 * @param image Source image data
 * @param maxWidthCells Maximum width in terminal cells
 * @param maxHeightCells Maximum height in terminal cells
 */
export async function scaleImageToFit(
  image: ImageData,
  maxWidthCells: number,
  maxHeightCells: number
): Promise<ImageData> {
  // Each cell is 1 pixel wide, 2 pixels tall (half-blocks)
  const maxWidthPx = maxWidthCells
  const maxHeightPx = maxHeightCells * 2

  // Calculate scale factor to fit
  const scaleX = maxWidthPx / image.width
  const scaleY = maxHeightPx / image.height
  const scale = Math.min(scaleX, scaleY, 1) // Don't upscale

  const newWidth = Math.max(1, Math.floor(image.width * scale))
  const newHeight = Math.max(2, Math.floor(image.height * scale))

  // If no scaling needed, return original
  if (newWidth === image.width && newHeight === image.height) {
    return image
  }

  // Use Jimp to resize
  try {
    const jimpImage = new Jimp({ data: Buffer.from(image.pixels), width: image.width, height: image.height })
    jimpImage.resize({ w: newWidth, h: newHeight })

    return {
      pixels: new Uint8Array(jimpImage.bitmap.data),
      width: jimpImage.bitmap.width,
      height: jimpImage.bitmap.height,
    }
  } catch (error) {
    console.error("Error scaling image:", error)
    return image // Return original on error
  }
}

/**
 * Get pixel color at (x, y) from image data.
 */
function getPixel(image: ImageData, x: number, y: number): RGBA {
  if (x < 0 || x >= image.width || y < 0 || y >= image.height) {
    return RGBA.fromInts(0, 0, 0, 0)
  }

  const index = (y * image.width + x) * 4
  return RGBA.fromInts(
    image.pixels[index],
    image.pixels[index + 1],
    image.pixels[index + 2],
    image.pixels[index + 3]
  )
}

/**
 * Check if two colors are similar enough to use a single color.
 */
function colorsSimilar(a: RGBA, b: RGBA, threshold: number = 0.1): boolean {
  const dr = Math.abs(a.r - b.r)
  const dg = Math.abs(a.g - b.g)
  const db = Math.abs(a.b - b.b)
  return dr < threshold && dg < threshold && db < threshold
}

/**
 * Render an image to an OptimizedBuffer using half-block characters.
 *
 * @param buffer Target buffer to render to
 * @param image Image data to render
 * @param x X position in terminal cells
 * @param y Y position in terminal cells
 * @param bgColor Background color to blend transparent pixels with
 */
export function renderImageToBuffer(
  buffer: OptimizedBuffer,
  image: ImageData,
  x: number,
  y: number,
  bgColor: RGBA = RGBA.fromInts(0, 0, 0, 255)
): void {
  // Calculate cell dimensions
  const cellWidth = image.width
  const cellHeight = Math.ceil(image.height / 2)

  for (let cy = 0; cy < cellHeight; cy++) {
    for (let cx = 0; cx < cellWidth; cx++) {
      const pixelY = cy * 2

      // Get top and bottom pixel colors
      const topPixel = getPixel(image, cx, pixelY)
      const bottomPixel = getPixel(image, cx, pixelY + 1)

      // Blend with background if transparent
      const topColor = topPixel.a < 0.5 ? bgColor : topPixel
      const bottomColor = bottomPixel.a < 0.5 ? bgColor : bottomPixel

      // Choose character and colors based on pixel values
      let char: string
      let fg: RGBA
      let bg: RGBA

      if (colorsSimilar(topColor, bottomColor)) {
        // Both pixels similar - use space with background color
        char = SPACE
        fg = topColor
        bg = topColor
      } else {
        // Different colors - use half block
        // Upper half block: foreground is top, background is bottom
        char = UPPER_HALF_BLOCK
        fg = topColor
        bg = bottomColor
      }

      // Draw the cell
      const cellX = x + cx
      const cellY = y + cy

      // Cast to any to access width/height properties (type definitions may be incomplete)
      const buf = buffer as any
      if (cellX >= 0 && cellX < buf.width && cellY >= 0 && cellY < buf.height) {
        buf.setCell(cellX, cellY, char, fg, bg)
      }
    }
  }
}

/**
 * Calculate the terminal cell dimensions for an image.
 */
export function getImageCellDimensions(image: ImageData): { width: number; height: number } {
  return {
    width: image.width,
    height: Math.ceil(image.height / 2),
  }
}

/**
 * Render an image using half-block characters.
 * This is a wrapper around renderImageToBuffer for the modal.
 *
 * @param buffer Target OptimizedBuffer
 * @param image Image data (RGBA format)
 * @param x X position in terminal cells
 * @param y Y position in terminal cells
 */
export function renderImageNative(
  buffer: OptimizedBuffer,
  image: ImageData,
  x: number,
  y: number
): void {
  // Use half-block rendering (1 pixel width, 2 pixels height per cell)
  renderImageToBuffer(buffer, image, x, y)
}

/**
 * Get the terminal cell dimensions for rendering.
 * Half-block rendering uses 1 pixel per cell width, 2 pixels per cell height.
 */
export function getImageCellDimensionsNative(image: ImageData): { width: number; height: number } {
  return {
    width: image.width,
    height: Math.ceil(image.height / 2),
  }
}
