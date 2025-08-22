import React, { useEffect, useMemo, useRef, useState } from 'react'

type Props = {
  text: string | string[]
  typingSpeed?: number        // ms per char
  pauseDuration?: number      // ms after finishing one item
  showCursor?: boolean
  cursorChar?: string
  className?: string
}

const TextType: React.FC<Props> = ({
  text,
  typingSpeed = 35,
  pauseDuration = 1600,
  showCursor = true,
  cursorChar = '|',
  className = 'tt-text',
}) => {
  const items = useMemo(() => (Array.isArray(text) ? text : [text]).filter(Boolean), [text])
  const [idx, setIdx] = useState(0)
  const [out, setOut] = useState('')
  const timer = useRef<number | null>(null)

  useEffect(() => {
    if (!items.length) return
    let mounted = true
    let i = 0
    const word = items[idx] ?? ''
    const tick = () => {
      if (!mounted) return
      if (i <= word.length) {
        setOut(word.slice(0, i))
        i += 1
        timer.current = window.setTimeout(tick, typingSpeed)
      } else {
        timer.current = window.setTimeout(() => {
          if (!mounted) return
          setIdx((idx + 1) % items.length)
        }, pauseDuration)
      }
    }
    tick()
    return () => {
      mounted = false
      if (timer.current) window.clearTimeout(timer.current)
    }
  }, [idx, items, typingSpeed, pauseDuration])

  return (
    <div className={className}>
      {out}
      {showCursor ? <span className="tt-cursor">{cursorChar}</span> : null}
    </div>
  )
}

export default TextType